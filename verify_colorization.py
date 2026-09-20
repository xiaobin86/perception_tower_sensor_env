#!/usr/bin/env python3
"""用相机自带的彩色点云（depth_cloud_colored.ply）检验雷达上色（colored.ply）是否正确。

原理：两个传感器看同一场景，同一块物理表面在两条点云上应拿到相同颜色。
做法：用外参把雷达点云变到相机系 → 对每个雷达点找最近相机点（几何贴合 <tol）→ 比较 RGB。
颜色差小（且与距离无关）= 外参 + 上色都对；差大或随距离增长 = 外参残差。

用法:
    python3 verify_colorization.py turntable_output/<DIR>
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree


def read_ply_xyzrgb(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path) as f:
        header, n_col = [], 0
        for i, line in enumerate(f, 1):
            header.append(line)
            if line.strip() == "end_header":
                break
        has_rgb = any("red" in l or "rgb" in l or "r" in l.split()[:1] for l in header)
    data = np.loadtxt(path, skiprows=i)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3]
    rgb = data[:, 3:6] if (has_rgb and data.shape[1] >= 6) else None
    return xyz, rgb


def rotation_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify LiDAR colorization against the camera's own colored cloud")
    parser.add_argument("pose_dir")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--geom-tol", type=float, default=0.05, help="几何贴合阈值 m")
    parser.add_argument("--max-dist", type=float, default=5.0)
    args = parser.parse_args()

    lidar_path = os.path.join(args.pose_dir, "colored.ply")
    cam_path = os.path.join(args.pose_dir, "depth_cloud_colored.ply")
    for p in (lidar_path, cam_path):
        if not os.path.exists(p):
            print(f"缺少文件: {p}")
            return 1

    ex = yaml.safe_load(open(args.extrinsics))["lidar_to_camera"]
    R = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t = np.array(ex["translation"])

    lidar_xyz, lidar_rgb = read_ply_xyzrgb(lidar_path)
    cam_xyz, cam_rgb = read_ply_xyzrgb(cam_path)
    if lidar_rgb is None or cam_rgb is None:
        print("缺少颜色列（colored.ply / depth_cloud_colored.ply 都需要 rgb）")
        return 1
    print(f"雷达彩色云 {len(lidar_xyz)} 点, 相机彩色云 {len(cam_xyz)} 点")

    Rz = rotation_z(args.photo_angle)
    lc = (R @ ((lidar_xyz @ Rz.T).T)).T + t
    gray = np.all(np.abs(lidar_rgb - 30.0) < 1.0, axis=1)
    keep = (~gray) & (lc[:, 2] > 0.3) & (lc[:, 2] < args.max_dist)
    lc, lrgb = lc[keep], lidar_rgb[keep]
    print(f"雷达已上色且深度有效: {len(lc)} 点")

    rng = np.random.default_rng(0)
    if len(cam_xyz) > 300000:
        sel = rng.choice(len(cam_xyz), 300000, replace=False)
        cam_xyz, cam_rgb = cam_xyz[sel], cam_rgb[sel]
    tree = cKDTree(cam_xyz)
    d, idx = tree.query(lc, k=1)
    matched = d < args.geom_tol
    if matched.sum() < 200:
        print(f"几何贴合点太少 ({int(matched.sum())})；外参可能偏太多")
        return 1
    diff = np.abs(lrgb[matched] - cam_rgb[idx[matched]]).mean(axis=1)
    z = lc[matched, 2]
    print(f"几何贴合 <{args.geom_tol*100:.0f}cm 的点: {int(matched.sum())} ({matched.mean():.0%})")
    print(f"颜色差 |ΔRGB|: 中位={np.median(diff):.1f}  均值={diff.mean():.1f}  <20 占比={np.mean(diff<20):.0%}  <40 占比={np.mean(diff<40):.0%}")
    print(f"{'距离段':<12}{'点数':>8}{'颜色差中位':>12}")
    for lo, hi in ((0.3, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.0)):
        sel = (z >= lo) & (z < hi)
        if sel.sum() < 100:
            continue
        print(f"{f'{lo}-{hi}m':<12}{int(sel.sum()):>8}{np.median(diff[sel]):>12.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
