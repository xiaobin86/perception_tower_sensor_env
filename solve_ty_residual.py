#!/usr/bin/env python3
"""用相机深度云联解 t.y 调整量与俯仰残差(只测量, 不应用)。

原理: 深度图 D(u,v) 预计算竖直梯度 dDdv。雷达点经当前外参投影, t.y 变化 Δ 使其
采样像素 v 移动 fy·Δ/Z, 残差线性化:
    r(δ, Δ) ≈ r0 + δ·g − Δ·h
    g = [a×p]_z (a=世界X在相机系, 俯仰贡献)
    h = (fy/Z)·dDdv          (t.y 贡献)
稳健最小二乘(3轮 MAD 重加权)解 (δ, Δ); Δ>0 = t.y 应再增大(点云上移)。

用法: python3 solve_ty_residual.py [帧目录...]   (默认全部 20260918_0*)
"""

import glob
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, load_extrinsics, rotation_z
from solve_pitch_residual import depth_image


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
    Zs = Zimg.copy()
    Zs[np.isnan(Zs)] = np.nanmedian(Zs[~np.isnan(Zs)])
    Zs = cv2.GaussianBlur(Zs.astype(np.float32), (9, 9), 0)
    dDdv = cv2.Sobel(Zs, cv2.CV_32F, 0, 1, ksize=5) / 16.0  # 深度/像素(v向)

    Rf = R @ rotation_z(photo)
    P = (Rf @ lid.T).T + t
    z = P[:, 2]
    ok = z > 0.4
    u = np.round(K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]).astype(int)
    m = (u >= 4) & (u < 1276) & (v >= 4) & (v < 716)
    u, v = u[m], v[m]
    zl = z[ok][m]
    D = Zimg[v, u]
    valid = np.isfinite(D)
    u, v, zl, D = u[valid], v[valid], zl[valid], D[valid]

    a = Rf @ np.array([1.0, 0, 0])
    g = (np.cross(np.broadcast_to(a, P[ok].shape), P[ok]))[m][valid][:, 2]
    h = K[1, 1] * dDdv[v, u] / zl
    r0 = zl - D
    w = np.abs(r0) < gate
    for _ in range(3):
        A = np.stack([g[w], -h[w]], axis=1)
        sol, *_ = np.linalg.lstsq(A, -r0[w], rcond=None)
        delta, dty = float(sol[0]), float(sol[1])
        r = r0 + delta * g - dty * h
        mad = np.median(np.abs(r[w] - np.median(r[w]))) + 1e-6
        w = np.abs(r) < max(0.03, 3.0 * mad)
        if w.sum() < 500:
            break
    return dict(pose=os.path.basename(d.rstrip("/")), delta=delta, dty=dty,
                n=int(w.sum()), rms=float(np.sqrt((r[w] ** 2).mean())))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=[])
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    a = ap.parse_args()
    dirs = a.dirs or sorted(glob.glob("turntable_output/20260918_0*"))[:-1]
    K, _ = load_camera_info("config/camera_info.yaml")
    R, t = load_extrinsics(a.extrinsics)
    rows = []
    for d in dirs:
        r = solve_frame(d, R, t, K, 90.0)
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['pose']}: 俯仰={np.degrees(r['delta']):+.2f}°  "
              f"Δt.y={r['dty'] * 1000:+6.1f}mm  内点{r['n']:6d}  RMS {r['rms'] * 1000:5.1f}mm")
    if not rows:
        print("无有效帧")
        return 1
    D = np.array([r["delta"] for r in rows])
    Y = np.array([r["dty"] for r in rows])
    print(f"\n跨 {len(rows)} 帧中位: 俯仰 = {np.degrees(np.median(D)):+.2f}°   "
          f"**Δt.y = {np.median(Y) * 1000:+.1f}mm**   "
          f"(Δt.y 散布 sigma={np.degrees(0) + Y.std() * 1000:.1f}mm)")
    print(f"残差相对当前生效外参 (config/camera_extrinsics.yaml) → 深度云建议 t.y 调整 ≈ "
          f"{np.median(Y) * 1000:+.1f}mm (相对标定值)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
