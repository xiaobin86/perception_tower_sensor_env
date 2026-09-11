#!/usr/bin/env python3
"""相机-LiDAR 手动对齐工具。

用法：
    python3 align_camera_lidar.py /workspace/turntable_output/20260907_181431

操作：
    q/a : tx +/-
    w/s : ty +/-
    e/d : tz +/-
    r/f : rx +/- (绕X轴旋转，度)
    t/g : ry +/- (绕Y轴旋转，度)
    y/h : rz +/- (绕Z轴旋转，度)
    +/- : 调整步长
    p   : 打印当前位姿
    S   : 保存外参到 config/camera_extrinsics.yaml
    ESC : 退出
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_ply(ply_path: str) -> np.ndarray | None:
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        return np.asarray(pcd.points)
    except Exception as e:
        print(f"加载PLY失败: {e}")
        return None


def load_ply_manual(ply_path: str) -> np.ndarray | None:
    try:
        points = []
        with open(ply_path, 'r') as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == 'end_header':
                        in_header = False
                else:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if points:
            return np.array(points, dtype=np.float32)
        return None
    except Exception as e:
        print(f"加载PLY失败: {e}")
        return None


def get_points(ply_path: str) -> np.ndarray | None:
    if HAS_OPEN3D:
        return load_ply(ply_path)
    return load_ply_manual(ply_path)


def rotation_matrix_from_euler(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    """投影3D点到2D图像。
    
    返回：
        uv: (N, 2) 像素坐标
        valid: (N,) bool 是否在图像范围内且z>0
    """
    points_cam = (R @ points.T + t.reshape(3, 1)).T
    z = points_cam[:, 2]
    visible = z > 0.1
    
    uv = np.zeros((len(points), 2))
    uv[visible] = (K @ points_cam[visible].T / z[visible]).T[:, :2]
    
    in_image = (
        (uv[:, 0] >= 0) & (uv[:, 0] < image_width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < image_height) &
        visible
    )
    
    return uv, in_image


def draw_projection(image: np.ndarray, points: np.ndarray, K: np.ndarray,
                    R: np.ndarray, t: np.ndarray, step_size: float) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    
    uv, valid = project_points(points, K, R, t, w, h)
    
    if np.any(valid):
        z = (R @ points[valid].T + t.reshape(3, 1)).T[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            colors = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)
            colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
        else:
            colors = np.full((len(z), 1, 3), 128, dtype=np.uint8)
        
        for (u, v), color in zip(uv[valid].astype(int), colors):
            cv2.circle(img, (u, v), 1, tuple(int(c) for c in color[0]), -1)
    
    # 显示当前参数
    text_lines = [
        f"tx={t[0]:.4f} ty={t[1]:.4f} tz={t[2]:.4f}",
        f"rx={np.degrees(np.arctan2(R[2,1], R[2,2])):.2f} "
        f"ry={np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2))):.2f} "
        f"rz={np.degrees(np.arctan2(R[1,0], R[0,0])):.2f}",
        f"step={step_size:.4f}  valid_points={valid.sum()}",
        "q/a:tx w/s:ty e/d:tz  r/f:rx t/g:ry y/h:rz",
        "+/-:step p:print S:save ESC:exit",
    ]
    
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    return img


def euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """从旋转矩阵提取欧拉角（度）。"""
    rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    ry = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return rx, ry, rz


def main():
    parser = argparse.ArgumentParser(description="相机-LiDAR手动对齐")
    parser.add_argument("data_dir", help="数据目录（包含merged.ply、color.png、config/camera_info.yaml）")
    args = parser.parse_args()
    
    merged_path = os.path.join(args.data_dir, "merged.ply")
    image_path = os.path.join(args.data_dir, "color.png")
    camera_info_path = os.path.join(args.data_dir, "config", "camera_info.yaml")
    
    if not os.path.exists(merged_path):
        print(f"错误: {merged_path} 不存在")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"错误: {image_path} 不存在")
        sys.exit(1)
    if not os.path.exists(camera_info_path):
        print(f"错误: {camera_info_path} 不存在")
        print("请先运行 save_camera_info.py 保存相机内参")
        sys.exit(1)
    
    points = get_points(merged_path)
    if points is None or len(points) == 0:
        print("错误: 点云为空")
        sys.exit(1)
    
    # 降采样，太多点投影太慢
    max_points = 100000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    image = cv2.imread(image_path)
    if image is None:
        print("错误: 无法读取图片")
        sys.exit(1)
    
    camera_info = load_yaml(camera_info_path)
    K = np.array(camera_info["k"]).reshape(3, 3)
    
    print(f"加载图像: {image.shape[1]}x{image.shape[0]}")
    print(f"加载点云: {len(points)} 点")
    print(f"相机内参 K:\n{K}")
    
    # 初始位姿（需要手动调整）
    t = np.array([0.0, 0.0, 1.0])
    R = np.eye(3)
    step_size = 0.01
    
    window_name = "Camera-LiDAR Alignment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\n操作说明：")
    print("q/a : tx +/-   w/s : ty +/-   e/d : tz +/-")
    print("r/f : rx +/-   t/g : ry +/-   y/h : rz +/-")
    print("+/- : 调整步长")
    print("p   : 打印当前位姿")
    print("S   : 保存外参")
    print("ESC : 退出\n")
    
    while True:
        img = draw_projection(image, points, K, R, t, step_size)
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('q'):
            t[0] += step_size
        elif key == ord('a'):
            t[0] -= step_size
        elif key == ord('w'):
            t[1] += step_size
        elif key == ord('s'):
            t[1] -= step_size
        elif key == ord('e'):
            t[2] += step_size
        elif key == ord('d'):
            t[2] -= step_size
        elif key == ord('r'):
            R = rotation_matrix_from_euler(step_size, 0, 0) @ R
        elif key == ord('f'):
            R = rotation_matrix_from_euler(-step_size, 0, 0) @ R
        elif key == ord('t'):
            R = rotation_matrix_from_euler(0, step_size, 0) @ R
        elif key == ord('g'):
            R = rotation_matrix_from_euler(0, -step_size, 0) @ R
        elif key == ord('y'):
            R = rotation_matrix_from_euler(0, 0, step_size) @ R
        elif key == ord('h'):
            R = rotation_matrix_from_euler(0, 0, -step_size) @ R
        elif key == ord('+'):
            step_size *= 2.0
        elif key == ord('-'):
            step_size /= 2.0
        elif key == ord('p'):
            rx, ry, rz = euler_from_matrix(R)
            print(f"当前位姿: t={t.tolist()}, rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}")
        elif key == ord('S'):
            output_dir = os.path.join(args.data_dir, "config")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "camera_extrinsics.yaml")
            
            rx, ry, rz = euler_from_matrix(R)
            data = {
                "translation": t.tolist(),
                "rotation_matrix": R.flatten().tolist(),
                "euler_deg": [rx, ry, rz],
            }
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"已保存外参: {output_path}")
    
    cv2.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    main()
