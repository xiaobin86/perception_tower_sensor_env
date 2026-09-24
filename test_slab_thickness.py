#!/usr/bin/env python3
"""实验: 板面厚度容差 DIST_THR 2cm vs 3cm。

动机: 黑格反光弱 -> 测距偏远 -> 被 ±2cm 平面容差剥离(点云格子镂空)。
方法: monkey-patch sb.DIST_THR, 不改动正式代码。对比窗口位置(相机PnP真板)、
占比、点数, 并把 3cm 结果的选中/带内点落地为 PLY 供目视。
"""

import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_board as sb
from calibrate_camera_lidar import (BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_X,
                                    SQUARE_SIZE_Y, board_object_points,
                                    find_board_corners, load_camera_info,
                                    solve_pnp_robust)

POSES = ["turntable_output/20260923_072321", "turntable_output/20260923_072547"]


def read_ply_xyz(path):
    with open(path) as f:
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        return np.loadtxt(f, max_rows=n)[:, :3].astype(np.float64)


def camera_truth(pose_dir):
    K, dist = load_camera_info("config/camera_info.yaml")
    img = cv2.imread(os.path.join(pose_dir, "color.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(
        gray, find_board_corners(img), (5, 5), (-1, -1), crit).reshape(-1, 2)
    rvec, tvec, *_ = solve_pnp_robust(board_object_points(), corners, K, dist)
    Rc, _ = cv2.Rodrigues(rvec)
    obj_c = np.array([(BOARD_COLS - 1) / 2 * SQUARE_SIZE_X,
                      (BOARD_ROWS - 1) / 2 * SQUARE_SIZE_Y, 0.0])
    center_cam = Rc @ obj_c + tvec.ravel()
    ext = yaml.safe_load(open("config/camera_extrinsics.yaml"))["lidar_to_camera"]
    R = np.array(ext["rotation_matrix"]).reshape(3, 3)
    t = np.array(ext["translation"])
    c_photo = R.T @ (center_cam - t)
    a = np.radians(-90.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ c_photo


def win_center_3d(info):
    th = np.radians(info["ang"])
    ca, sa = np.cos(th), np.sin(th)
    A = np.array([[ca, sa], [-sa, ca]])
    u0, v0 = np.linalg.solve(A, np.array([info["wc"], info["hc"]]))
    return info["c"] + u0 * info["e1"] + v0 * info["e2"]


def write_ply_rgb(path, pts, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def compare_pose(pose_dir):
    name = os.path.basename(pose_dir)
    print(f"\n=== {name} ===")
    P = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))
    truth = camera_truth(pose_dir)
    print(f"  真板中心(相机PnP): {truth.round(3)}")

    for thr in (0.02, 0.03):
        sb.DIST_THR = thr
        mask, info = sb.extract_rect_plane(P)
        sb.DIST_THR = 0.02
        if mask is None:
            print(f"  thr={thr*100:.0f}cm: 未找到")
            continue
        c3d = win_center_3d(info)
        d = np.linalg.norm(c3d - truth)
        band_n = int(info.get("band_cc_n", 0))
        ratio = info["n_sel"] / max(band_n, 1)
        print(f"  thr={thr*100:.0f}cm: win内点(降采样)={info['cnt']}  带CC={band_n}  "
              f"占比={ratio:.3f}  窗口距真板={d*100:.1f}cm  ang={info['ang']:.0f}  "
              f"n={info['n'].round(3)}")
        if thr == 0.03:
            # 落地可视化: 绿=选中, 红=带内未选, 灰=其他(mask 是全量云的)
            colors = np.full((len(P), 3), 200, np.uint8)
            band = info.get("band_mask")
            if band is not None and len(band) == len(P):
                colors[band] = (255, 0, 0)
            colors[mask] = (0, 200, 0)
            out = os.path.join(pose_dir, "slab3cm_debug.ply")
            write_ply_rgb(out, P, colors)
            print(f"    -> {out}")


if __name__ == "__main__":
    for pose in POSES:
        compare_pose(pose)
