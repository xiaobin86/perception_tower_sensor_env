#!/usr/bin/env python3
"""用相机深度云做图像域对齐微调（求外参平移补偿）。

为什么不用 3D 最近邻: 场景大面积平面 + 部分重叠时, 最近邻常配到别的表面上,
目标函数失效（实测 NN 中位 175mm、Δ 撞搜索边界）。

本法: 把相机云拍成稠密深度图 D_cam(u,v); 对每个雷达点(当前外参变换后):
        投影 → 取 D_cam 与局部梯度 (gx,gy)
        最小二乘   D_lidar − D_cam ≈ gx·du + gy·dv
      → (du,dv) = 雷达云在图像中需要移动的像素量
      → 平移补偿  ΔX = du·Z/fx, ΔY = dv·Z/fy （Z 用中位深度）
      方向由数据决定, 无需猜符号。

用法:
    python3 solve_shift_by_depth.py <pose_dir> [<pose_dir> ...]
        [--extrinsics config/camera_extrinsics.yaml] [--photo-angle 90] [--iters 3]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt, sobel

from verify_colorization import read_ply_xyzrgb, rotation_z
from colorize_pointcloud import load_camera_info as _load_K


def load_camera_info(path: str) -> tuple[np.ndarray, int, int]:
    K, _ = _load_K(path)
    d = yaml.safe_load(open(path))
    return K, int(d["height"]), int(d["width"])


def dense_depth(pts: np.ndarray, K: np.ndarray, H: int, W: int) -> np.ndarray:
    z = pts[:, 2]
    ok = z > 0.25
    u = np.round(K[0, 0] * pts[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pts[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
    m = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < 8.0)
    img = np.full((H, W), np.inf)
    np.minimum.at(img, (v[m], u[m]), z[m])
    img[~np.isfinite(img)] = np.nan
    nan = np.isnan(img)
    if nan.any():
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        img = img[tuple(idx)]
    return img


def solve_pose(pose_dir: str, ex_path: str, info_path: str, photo_angle: float,
               iters: int) -> dict | None:
    lp = os.path.join(pose_dir, "colored.ply")
    cp = os.path.join(pose_dir, "depth_cloud_colored.ply")
    if not (os.path.exists(lp) and os.path.exists(cp)):
        print(f"  skip {os.path.basename(pose_dir)}: 缺文件")
        return None
    K, H, W = load_camera_info(info_path)
    ex = yaml.safe_load(open(ex_path))["lidar_to_camera"]
    R = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t = np.array(ex["translation"])

    lidar, _ = read_ply_xyzrgb(lp)
    cam, _ = read_ply_xyzrgb(cp)
    Dcam = dense_depth(cam, K, H, W)
    gx = sobel(Dcam, axis=1, mode="nearest")
    gy = sobel(Dcam, axis=0, mode="nearest")

    P = (R @ ((lidar @ rotation_z(photo_angle).T).T)).T + t
    z = P[:, 2]
    u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    inside = (z > 0.4) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
    print(f"    画面内雷达点: {int(inside.sum())}/{len(P)}")
    u, v, zz = u[inside], v[inside], z[inside]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    A = np.column_stack([gx[vi, ui], gy[vi, ui]])
    b = zz - Dcam[vi, ui]
    ok = np.isfinite(A).all(1) & np.isfinite(b) & (np.abs(A) < 100.0).any(1)
    A, b, zz = A[ok], b[ok], zz[ok]
    print(f"    参与拟合: {len(A)} 点, |D差|中位 {np.median(np.abs(b))*1000:.1f}mm")

    du = dv = 0.0
    for _ in range(iters):
        sel = np.abs(b - A @ [du, dv]) < 0.05
        if sel.sum() < 500:
            break
        sol, *_ = np.linalg.lstsq(A[sel], b[sel], rcond=None)
        du, dv = float(sol[0]), float(sol[1])
    resid = A @ [du, dv] - b
    med_z = float(np.median(zz))
    dx = du * med_z / K[0, 0]
    dy = dv * med_z / K[1, 1]
    return {"pose": os.path.basename(pose_dir), "du": du, "dv": dv,
            "dx": dx, "dy": dy, "z": med_z,
            "n": int(sel.sum()), "rms": float(np.sqrt((resid**2).mean()))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Solve extrinsic shift by camera-depth image alignment")
    ap.add_argument("pose_dirs", nargs="+")
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--camera-info", default="config/camera_info.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()

    out = []
    for p in a.pose_dirs:
        r = solve_pose(p, a.extrinsics, a.camera_info, a.photo_angle, a.iters)
        if r is None:
            continue
        out.append(r)
        print(f"{r['pose']}: du={r['du']:+6.2f}px dv={r['dv']:+6.2f}px "
              f"→ ΔX={r['dx']*1000:+6.1f}mm ΔY={r['dy']*1000:+6.1f}mm "
              f"(中位深度 {r['z']:.2f}m, {r['n']} 点, 残差 {r['rms']*1000:.1f}mm)")
    if not out:
        print("无有效位姿")
        return 1
    dx = np.array([r["dx"] for r in out])
    dy = np.array([r["dy"] for r in out])
    print(f"\n跨 {len(out)} 帧共识: ΔX = {np.median(dx)*1000:+.1f} mm (散布 {dx.std()*1000:.1f})  "
          f"ΔY = {np.median(dy)*1000:+.1f} mm (散布 {dy.std()*1000:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
