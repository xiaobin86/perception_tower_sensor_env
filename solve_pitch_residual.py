#!/usr/bin/env python3
"""用相机深度云解外参残差: 绕世界X的微小旋转(用户观察到的) + t_z。

原理: 深度云(相机系)拍成 Z 图 D(u,v) = 独立基准。雷达点经当前外参投影后采样深度,
残差 r = Z_lidar - D_cam 对绕世界X的小旋转 delta 线性:
    Z(delta) = Z + delta * [a x p]_z ,  a = R @ x_world (世界X在相机系的像)
稳健最小二乘解 delta(迭代 3 轮剔粗差); 顺带输出最优 t_z 平移。
解出后: R <- R @ Rx(delta)(在合并系侧修正), t_z 相应平移。

用法: python3 solve_pitch_residual.py <帧目录...> [--extrinsics ...] [--apply]
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, load_extrinsics, rotation_z


def depth_image(pts_cam, K, H=720, W=1280):
    """相机系点云 -> Z 图(每像素取最近值)。"""
    Z = np.full((H, W), np.nan)
    z = pts_cam[:, 2]
    ok = z > 0.3
    u = np.round(K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(int)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[m], v[m], z[ok][m]
    order = np.argsort(z)[::-1]           # 远的先写, 近的覆盖
    Z[v[order], u[order]] = z[order]
    return Z


def solve_frame(d, R, t, K, photo, gate=0.25):
    dc = os.path.join(d, "depth_cloud.ply")
    mp = os.path.join(d, "merged.ply")
    if not (os.path.exists(dc) and os.path.exists(mp)):
        return None
    with open(dc) as f:
        for i, l in enumerate(f, 1):
            if l.strip() == "end_header":
                break
    cam = np.loadtxt(dc, skiprows=i)[:, :3]
    with open(mp) as f:
        for i, l in enumerate(f, 1):
            if l.strip() == "end_header":
                break
    lid = np.loadtxt(mp, skiprows=i)[:, :3]
    rng = np.random.default_rng(0)
    if len(lid) > 150000:
        lid = lid[rng.choice(len(lid), 150000, replace=False)]
    Zimg = depth_image(cam, K)
    Rf = R @ rotation_z(photo)
    P = (Rf @ lid.T).T + t
    z = P[:, 2]
    ok = z > 0.4
    u = np.round(K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]).astype(int)
    m = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    u, v = u[m], v[m]
    zl = z[ok][m]
    D = Zimg[v, u]
    valid = np.isfinite(D)
    u, v, zl, D = u[valid], v[valid], zl[valid], D[valid]

    a = Rf @ np.array([1.0, 0, 0])          # 世界X在相机系
    g = (np.cross(np.broadcast_to(a, P[ok].shape), P[ok]))[m][valid][:, 2]
    r0 = zl - D
    w = np.abs(r0) < gate
    for _ in range(3):
        A = g[w]
        b = -r0[w]
        delta = float(A @ b / max(A @ A, 1e-9))
        r = r0 + delta * g
        mad = np.median(np.abs(r[w] - np.median(r[w]))) + 1e-6
        w = np.abs(r) < max(0.03, 3.0 * mad)
        if w.sum() < 500:
            break
    r_final = r0 + delta * g
    dz = float(np.median(r_final[w]))
    n_in = int(w.sum())
    return dict(pose=os.path.basename(d.rstrip("/")), delta=delta, dz=dz, n=n_in,
                rms=float(np.sqrt((r_final[w] ** 2).mean())))


def main() -> int:
    ap = argparse.ArgumentParser(description="solve world-X pitch residual from camera depth cloud")
    ap.add_argument("dirs", nargs="*", default=[])
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--camera-info", default="config/camera_info.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--apply", action="store_true", help="解出后写入新外参(归档旧版)")
    a = ap.parse_args()
    dirs = a.dirs or sorted(glob.glob("turntable_output/20260918_0*"))[:-1]
    K, _ = load_camera_info(a.camera_info)
    R, t = load_extrinsics(a.extrinsics)

    rows = []
    for d in dirs:
        r = solve_frame(d, R, t, K, a.photo_angle)
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['pose']}: delta={np.degrees(r['delta']):+.3f}°  "
              f"Δtz={r['dz'] * 1000:+.1f}mm  内点{r['n']:6d}  RMS {r['rms'] * 1000:5.1f}mm")
    if not rows:
        print("无有效帧")
        return 1
    D = np.array([r["delta"] for r in rows])
    Z = np.array([r["dz"] for r in rows])
    d_med = float(np.median(D))
    z_med = float(np.median(Z))
    print(f"\n跨 {len(rows)} 帧中位: 绕世界X旋转 = {np.degrees(d_med):+.3f}°   "
          f"Δtz = {z_med * 1000:+.1f}mm   (散布 sigma={np.degrees(D.std()):.3f}°)")

    if a.apply:
        c = np.cos(d_med)
        s = np.sin(d_med)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        Rn = R @ Rx
        tn = t + np.array([0, 0, z_med])
        import shutil, datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(a.extrinsics, f"config/extrinsics_history/{stamp}_before_pitch_fix.yaml")
        yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": Rn.flatten().tolist(),
                                            "translation": tn.tolist()}},
                       open(a.extrinsics, "w"), sort_keys=False)
        print(f"\n已应用: R <- R·Rx({np.degrees(d_med):+.3f}°), t.z += {z_med * 1000:+.1f}mm")
        print(f"  新 t = {np.round(tn, 5).tolist()}   旧版已归档")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
