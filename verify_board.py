#!/usr/bin/env python3
"""验证 segment_board 分割结果: 把板投影到照片, 与图像检出的棋盘格外框算重合率。

重合率 = 落在图像棋盘格四边形内的投影点比例 (>70% 确认 / 30~70% 可疑 / <30% 错误)。
相机是独立基准 —— 这就是"以图像为准"的客观判决。
用法: python3 verify_board.py [帧目录...]  (默认全部 20260918_*)
"""

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_board as sb
from calibrate_camera_lidar import find_board_corners, board_object_points
from colorize_pointcloud import load_camera_info, load_extrinsics, rotation_z


def main() -> int:
    K, dist = load_camera_info("config/camera_info.yaml")
    R, t = load_extrinsics("config/camera_extrinsics.yaml")
    objp = board_object_points()
    dirs = sys.argv[1:] or sorted(glob.glob("turntable_output/20260918_0*"))[:-1]

    print(f"{'帧':<12}{'分割点数':>8}{'重合率':>8}  判定")
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        with open(os.path.join(d, "merged.ply")) as f:
            for i, line in enumerate(f, 1):
                if line.strip() == "end_header":
                    break
        P = np.loadtxt(os.path.join(d, "merged.ply"), skiprows=i)[:, :3]
        mask, info = sb.extract_rect_plane(P)
        if mask is None:
            print(f"  {name:<10}{'--':>8}{'--':>8}  分割失败")
            continue
        sel = P[mask]
        im = cv2.imread(os.path.join(d, "color.png"))
        try:
            corners = find_board_corners(im)
        except Exception:
            print(f"  {name:<10}{len(sel):>8}{'--':>8}  无相机基准(角点检不出), 需目检")
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        corners = corners.reshape(-1, 2)
        _, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        gc = np.array([0.18, 0.30])
        half = np.array([0.30, 0.42])
        b4 = np.array([[gc[0] + sx * half[0], gc[1] + sy * half[1], 0.0]
                       for sx in (-1, 1) for sy in (-1, 1)])
        quad, _ = cv2.projectPoints(b4, rvec, tvec, K, dist)
        quad = cv2.convexHull(quad.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
        Rf = R @ rotation_z(90.0)
        uv, _ = cv2.projectPoints(sel, cv2.Rodrigues(Rf)[0], t, K, dist)
        uv = uv.reshape(-1, 2)
        H, W = im.shape[:2]
        m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        uv = uv[m]
        if len(uv) == 0:
            print(f"  {name:<10}{len(sel):>8}{'0%':>8}  错误(投影全在画面外)")
            continue
        q32 = quad.astype(np.float32)
        inq = np.array([cv2.pointPolygonTest(q32, (float(u), float(v)), False) >= 0
                        for u, v in uv])
        ov = inq.mean()
        tag = "确认" if ov > 0.7 else ("可疑" if ov > 0.3 else "错误")
        print(f"  {name:<10}{len(sel):>8}{ov * 100:>7.0f}%  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
