"""雷达安装方式配置：将"安装姿态"从代码中解耦为可配置数据。

安装方式 = 两个独立的变换，对应两处安装自由度：
  1. mount（棱镜 0° 参考相位）：LakiBeam 出厂 0° 参考与雷达 x 轴存在夹角，
     安装时绕雷达某个轴旋转对齐（通常绕 z 轴 90°）。
  2. to_world（安装姿态）：雷达系三个轴相对世界系（转盘系）的指向，
     用"世界 x/y/z 各取自雷达系哪个轴（可带符号）"描述。

换安装方式（正装/横装/倾斜）时只需改 YAML 配置文件，
不需要改任何代码。默认内置 side-mount（横装）配置。

YAML 格式示例（configs/install_side_mount.yaml）：
    name: side-mount
    description: 横装：雷达 x 下、y 左、z 前
    mount:
      axis: z          # 棱镜 0° 参考相位绕雷达系哪个轴旋转
      angle_deg: 90.0  # 旋转角度
    to_world:
      x: z             # 世界 x 轴 = 雷达 z 轴（前）
      y: y             # 世界 y 轴 = 雷达 y 轴（左）
      z: -x            # 世界 z 轴 = -雷达 x 轴（下→上）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# 雷达系三个单位轴向量（x/y/z）
_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _axis_vector(axis: str) -> np.ndarray:
    """解析形如 'x' / '-x' / 'z' 的轴名，返回带符号的单位向量。"""
    axis = axis.strip().lower()
    sign = 1.0
    if axis.startswith("-"):
        sign = -1.0
        axis = axis[1:]
    if axis not in _AXES:
        raise ValueError(f"无效的轴名: {axis!r}（应为 x/y/z 或 -x/-y/-z）")
    return sign * _AXES[axis]


def _axis_name(vec: np.ndarray) -> str:
    """单位向量 → 轴名（含符号），用于默认配置展示。"""
    for name, base in _AXES.items():
        if np.allclose(vec, base):
            return name
        if np.allclose(vec, -base):
            return "-" + name
    raise ValueError(f"非坐标轴方向: {vec}")


@dataclass
class InstallConfig:
    """一次完整的雷达安装方式描述。"""

    name: str = "side-mount"
    description: str = "横装：雷达 x 下、y 左、z 前（默认）"
    # 棱镜 0° 参考相位：绕雷达系 axis 旋转 angle_deg 度
    mount_axis: str = "z"
    mount_angle_deg: float = 90.0
    # 世界系 x/y/z 三轴分别取自雷达系哪个轴（可带负号）
    world_x: str = "z"
    world_y: str = "y"
    world_z: str = "-x"
    # 转盘旋转轴（世界系），拼接聚合时绕它旋转
    turntable_axis: str = "z"
    # 安装后微小倾斜角（度），由 360° 自标定得到。
    # 顺序：Rx(roll) -> Ry(pitch) -> Rz(yaw)，即 R_tilt = Rz @ Ry @ Rx。
    tilt_roll_deg: float = 0.0
    tilt_pitch_deg: float = 0.0
    tilt_yaw_deg: float = 0.0
    # LiDAR 安装姿态相对理想横装的偏差（度）。
    # 顺序：Rx(roll) -> Ry(pitch) -> Rz(yaw)，即 R_lidar = Rz @ Ry @ Rx。
    lidar_tilt_roll_deg: float = 0.0
    lidar_tilt_pitch_deg: float = 0.0
    lidar_tilt_yaw_deg: float = 0.0
    # 光心相对转盘轴心的偏心校正（米，雷达系 y/z 方向）
    offset_y_m: float = 0.0
    offset_z_m: float = 0.0

    # ---- 构造 ----
    @classmethod
    def side_mount(cls) -> "InstallConfig":
        """横装（默认）：x 下、y 左、z 前，棱镜相位绕 z 90°。"""
        return cls()

    @classmethod
    def upright(cls) -> "InstallConfig":
        """正装（出厂姿态）：x 前、y 左、z 上，棱镜相位 0°。"""
        return cls(
            name="upright",
            description="正装（出厂姿态）：雷达 x 前、y 左、z 上",
            mount_axis="z",
            mount_angle_deg=0.0,
            world_x="x",
            world_y="y",
            world_z="z",
            turntable_axis="z",
        )

    # ---- 变换矩阵 ----
    def mount_matrix(self) -> np.ndarray:
        """棱镜 0° 参考相位旋转矩阵（绕雷达系 self.mount_axis 轴）。"""
        axis = self.mount_axis.strip().lower()
        sign = -1.0 if axis.startswith("-") else 1.0
        axis = axis.lstrip("-")
        if axis not in _AXES:
            raise ValueError(f"无效的安装旋转轴: {self.mount_axis!r}")
        return rotation_matrix(axis, sign * self.mount_angle_deg)

    def to_world_matrix(self) -> np.ndarray:
        """雷达系 → 世界系（安装姿态）变换矩阵。

        矩阵列为世界 x/y/z 三轴在雷达系的坐标：
        w = (p·x̂_w, p·ŷ_w, p·ẑ_w) = M^T·p，行向量形式 p @ M。
        """
        return np.column_stack([
            _axis_vector(self.world_x),
            _axis_vector(self.world_y),
            _axis_vector(self.world_z),
        ])

    def mount_transform(self, points: np.ndarray) -> np.ndarray:
        """棱镜相位：points @ mount_matrix().T。"""
        return points @ self.mount_matrix().T

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """安装姿态：雷达系点 → 世界系点（行向量 p @ M）。"""
        return points @ self.to_world_matrix()

    def tilt_matrix(self) -> np.ndarray:
        """安装后微小倾斜修正矩阵：R_tilt = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
        rx = rotation_matrix("x", self.tilt_roll_deg)
        ry = rotation_matrix("y", self.tilt_pitch_deg)
        rz = rotation_matrix("z", self.tilt_yaw_deg)
        return rz @ ry @ rx

    def lidar_tilt_matrix(self) -> np.ndarray:
        """LiDAR 安装偏差矩阵：R_lidar = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
        rx = rotation_matrix("x", self.lidar_tilt_roll_deg)
        ry = rotation_matrix("y", self.lidar_tilt_pitch_deg)
        rz = rotation_matrix("z", self.lidar_tilt_yaw_deg)
        return rz @ ry @ rx

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "mount": {"axis": self.mount_axis, "angle_deg": self.mount_angle_deg},
            "to_world": {"x": self.world_x, "y": self.world_y, "z": self.world_z},
            "turntable_axis": self.turntable_axis,
            "tilt": {
                "roll_deg": float(self.tilt_roll_deg),
                "pitch_deg": float(self.tilt_pitch_deg),
                "yaw_deg": float(self.tilt_yaw_deg),
            },
            "lidar_tilt": {
                "roll_deg": float(self.lidar_tilt_roll_deg),
                "pitch_deg": float(self.lidar_tilt_pitch_deg),
                "yaw_deg": float(self.lidar_tilt_yaw_deg),
            },
            "offset": {
                "y_m": float(self.offset_y_m),
                "z_m": float(self.offset_z_m),
            },
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "InstallConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"配置文件格式错误（应为 YAML 映射）: {path}")
        mount = data.get("mount", {})
        to_world = data.get("to_world", {})
        tilt = data.get("tilt", {})
        lidar_tilt = data.get("lidar_tilt", {})
        offset = data.get("offset", {})
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            mount_axis=mount.get("axis", "z"),
            mount_angle_deg=float(mount.get("angle_deg", 90.0)),
            world_x=to_world.get("x", "z"),
            world_y=to_world.get("y", "y"),
            world_z=to_world.get("z", "-x"),
            turntable_axis=data.get("turntable_axis", "z"),
            tilt_roll_deg=float(tilt.get("roll_deg", 0.0)),
            tilt_pitch_deg=float(tilt.get("pitch_deg", 0.0)),
            tilt_yaw_deg=float(tilt.get("yaw_deg", 0.0)),
            lidar_tilt_roll_deg=float(lidar_tilt.get("roll_deg", 0.0)),
            lidar_tilt_pitch_deg=float(lidar_tilt.get("pitch_deg", 0.0)),
            lidar_tilt_yaw_deg=float(lidar_tilt.get("yaw_deg", 0.0)),
            offset_y_m=float(offset.get("y_m", 0.0)),
            offset_z_m=float(offset.get("z_m", 0.0)),
        )


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """绕指定轴旋转 angle_deg 度的 3x3 矩阵（右手系）。"""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c, -s],
                         [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]])
    raise ValueError(f"不支持的旋转轴: {axis}（可选 x/y/z）")
