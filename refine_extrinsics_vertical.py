#!/usr/bin/env python3
"""用深度点云微调外参的竖直平移分量（相机系 y，即图像竖直方向）。

只优化 1 个参数：Δy_cam，判据 = 变换后的激光点云到深度云的最近邻中位距离（多帧合并统计）。
水平分量不做优化，避免把 Orbbec 深度/彩色相机基线的偏差吸收进外参。

用法:
    python3 refine_extrinsics_vertical.py                    # 扫描并打印最优值（不写文件）
    python3 refine_extrinsics_vertical.py --write             # 写回 config/camera_extrinsics.yaml（先备份）
    python3 refine_extrinsics_vertical.py --range 0.06 --step 0.002 --scans 6
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree


def load_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path}: no end_header")
    return np.loadtxt(path, skiprows=i, dtype=np.float64)[:, :3]


def load_extrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    ex = yaml.safe_load(open(path))["lidar_to_camera"]
    return np.array(ex["rotation_matrix"]).reshape(3, 3), np.array(ex["translation"])


def rotation_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the vertical translation of the extrinsics")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--data-root", default="turntable_output")
    parser.add_argument("--scans", type=int, default=6, help="用多少帧统计（越多越稳）")
    parser.add_argument("--range", type=float, default=0.06, help="搜索范围 ±m")
    parser.add_argument("--step", type=float, default=0.002, help="步长 m")
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--write", action="store_true", help="把最优修正写回外参文件（自动备份）")
    args = parser.parse_args()

    K = np.array(yaml.safe_load(open(args.camera_info))["k"]).reshape(3, 3)
    dist = np.array(yaml.safe_load(open(args.camera_info))["d"]).reshape(-1, 1)
    R, t = load_extrinsics(args.extrinsics)
    Rz = rotation_z(args.photo_angle)
    rvec = cv2.Rodrigues(R @ Rz)[0]

    dirs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.exists(os.path.join(args.data_root, d, "merged.ply"))
                  and os.path.exists(os.path.join(args.data_root, d, "depth_cloud.ply")))
    if not dirs:
        print("no scan with merged.ply + depth_cloud.ply", file=sys.stderr)
        return 1
    step = max(1, len(dirs) // args.scans)
    dirs = dirs[::step][:args.scans]

    samples = []
    for name in dirs:
        p = os.path.join(args.data_root, name)
        merged = load_ply_xyz(os.path.join(p, "merged.ply"))
        depth = load_ply_xyz(os.path.join(p, "depth_cloud.ply"))
        depth = depth[np.linalg.norm(depth, axis=1) < 8.0]
        if len(depth) < 1000:
            continue
        img = cv2.imread(os.path.join(p, "color.png"))
        h, w = img.shape[:2]
        pc = (R @ ((merged @ Rz.T).T)).T + t
        proj, _ = cv2.projectPoints(merged, rvec, t, K, dist)
        proj = proj.reshape(-1, 2)
        keep = ((pc[:, 2] > 0.2) & np.isfinite(proj).all(axis=1)
                & (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
                & (np.linalg.norm(merged, axis=1) < 8.0))
        idx = np.where(keep)[0]
        if len(idx) < 2000:
            continue
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, min(6000, len(idx)), replace=False)
        samples.append((pc[idx].copy(), cKDTree(depth)))
        print(f"  统计帧 {name}: {len(idx)} 点")

    if not samples:
        print("no usable scan", file=sys.stderr)
        return 1
    print(f"\n{'Δy_cam(mm)':>10} {'中位NN(mm)':>11} {'<2cm':>7}")
    best = None
    for dy in np.arange(-args.range, args.range + args.step / 2, args.step):
        med, p2 = [], []
        for pc, tree in samples:
            shifted = pc.copy()
            shifted[:, 1] += dy
            d, _ = tree.query(shifted, k=1)
            med.append(np.median(d))
            p2.append(np.mean(d < 0.02))
        med_m, p2_m = float(np.mean(med)) * 1000, float(np.mean(p2))
        mark = ""
        if best is None or med_m < best[0]:
            best = (med_m, dy, p2_m)
            mark = "  ←"
        print(f"{dy*1000:>10.0f} {med_m:>11.1f} {p2_m:>6.0%}{mark}")

    print(f"\n最优 Δy_cam = {best[1]*1000:+.0f} mm  (中位NN {best[0]:.1f}mm, <2cm {best[2]:.0%})")
    if args.write:
        t_new = t.copy()
        t_new[1] += best[1]
        shutil.copy(args.extrinsics, args.extrinsics + ".bak")
        with open(args.extrinsics, "w") as f:
            yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                                "translation": t_new.tolist()}}, f, sort_keys=False)
        print(f"已写回 {args.extrinsics}（备份 {args.extrinsics}.bak）")
    else:
        print("（未写文件；加 --write 应用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
