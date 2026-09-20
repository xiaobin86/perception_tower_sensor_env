#!/usr/bin/env python3
"""深度偏差诊断: 逐帧对比三个独立测距, 定位 tz 系统偏差的来源。

每帧输出:
  Z_pnp    相机PnP解出的板中心深度(依赖内参fy, 外参t不参与)
  Z_depth  深度云在板框像素处的Z中位(Orbbec深度, 独立传感器, 相机原点)
  Δ_cam    = Z_pnp - Z_depth  → 相机侧尺度/fy偏置检测(同原点同物点, 无需外参)
  r_lidar  雷达 merged 系中板中心距离(雷达测距尺度, 从LiDAR原点)

判读: Δ_cam 稳定非零 → 相机fy尺度问题; ≈0 而 tz 仍需微调 → 偏置在雷达测距/合并几何侧。

用法: python3 solve_range_bias.py <帧目录...>
"""

import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_camera_lidar import detect_camera_plane, load_camera_info, read_ply_xyz
from segment_board import extract_rect_plane
from solve_pitch_residual import depth_image


def main() -> int:
    dirs = sys.argv[1:]
    if not dirs:
        print("用法: python3 solve_range_bias.py <帧目录...>")
        return 1
    K, dist = load_camera_info("config/camera_info.yaml")
    print(f"{'帧':<18}{'Z_pnp':>8}{'Z_depth':>9}{'Δ_cam':>8}{'r_lidar':>9}")
    rows = []
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        img = cv2.imread(os.path.join(d, "color.png"))
        dc = os.path.join(d, "depth_cloud.ply")
        if img is None or not os.path.exists(dc):
            print(f"{name:<18} 缺 color.png/depth_cloud.ply, 跳过")
            continue
        plane, corners, _up, center_cam, polygon = detect_camera_plane(img, K, dist)
        Z_pnp = float(center_cam[2])
        with open(dc) as f:
            for i, l in enumerate(f, 1):
                if l.strip() == "end_header":
                    break
        cam = np.loadtxt(dc, skiprows=i)[:, :3]
        Zimg = depth_image(cam, K)
        mask = np.zeros(Zimg.shape, np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        vals = Zimg[(mask > 0) & np.isfinite(Zimg)]
        if len(vals) < 50:
            print(f"{name:<18} 板框内深度点不足, 跳过")
            continue
        Z_depth = float(np.median(vals))
        P = read_ply_xyz(os.path.join(d, "merged.ply"))
        m, info = extract_rect_plane(P)
        if m is None:
            print(f"{name:<18} 板分割失败, 跳过")
            continue
        r_lidar = float(np.linalg.norm(np.median(P[m], axis=0)))
        rows.append((name, Z_pnp, Z_depth, Z_pnp - Z_depth, r_lidar))
        print(f"{name:<18}{Z_pnp:>8.3f}{Z_depth:>9.3f}{Z_pnp - Z_depth:>+8.4f}{r_lidar:>9.3f}")
    if rows:
        D = np.array([r[3] for r in rows])
        print(f"\nΔ_cam 跨 {len(rows)} 帧: 中位 {np.median(D)*1000:+.1f}mm  "
              f"均值 {D.mean()*1000:+.1f}mm  散布 σ={D.std()*1000:.1f}mm")
        print("判读: |中位| > 10mm → 相机fy尺度偏置(占tz微调大头); ≈0 → 偏置在雷达侧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
