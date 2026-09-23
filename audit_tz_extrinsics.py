#!/usr/bin/env python3
"""用相机深度云仲裁两组候选外参的 t.z（只测量, 不应用）。

原理: 雷达点云经候选外参投影到相机系采样深度图 D(u,v), 残差 r = z_lidar - D_depth。
对纯 t.z 平移, r 的均值即 -Δt.z 误差(像素漂移对 D 的影响是二阶小量);
深度云是独立传感器, 不经外参 t, 是标定后的独立审计(判据来源见 AGENTS.md 选版逻辑第3条)。

用法: python3 audit_tz_extrinsics.py [帧目录...]   (默认全部 20260923_*)
"""

import glob
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, rotation_z
from solve_pitch_residual import depth_image

# 候选外参: (名称, R, t, photo_angle_deg)
# 官方 0923 解: 照片位姿参考系(photo=90)
R_OURS = np.array([[-0.0456956, -0.9989018, 0.0103492],
                   [-0.5753242, 0.0178465, -0.8177307],
                   [0.8166480, -0.0433209, -0.5755079]])
T_OURS = np.array([0.0263856, 0.0733844, 0.0317394])
# MATLAB lidarCameraTform: 角度0参考系(photo=0)
R_MLAB = np.array([[-0.9990, 0.0425, 0.0135],
                   [0.0125, 0.5581, -0.8297],
                   [-0.0427, -0.8287, -0.5581]])
T_MLAB = np.array([0.0321, 0.0508, -0.0780])


def load_ply_xyz(path):
    with open(path) as f:
        for i, l in enumerate(f, 1):
            if l.strip() == "end_header":
                break
    return np.loadtxt(path, skiprows=i)[:, :3]


def eval_frame(d, R, t, photo, K, gate=0.25):
    dc, mp = os.path.join(d, "depth_cloud.ply"), os.path.join(d, "merged.ply")
    if not (os.path.exists(dc) and os.path.exists(mp)):
        return None
    cam = load_ply_xyz(dc)
    lid = load_ply_xyz(mp)
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
    m = (u >= 4) & (u < 1276) & (v >= 4) & (v < 716)
    u, v, zl = u[m], v[m], z[ok][m]
    D = Zimg[v, u]
    valid = np.isfinite(D)
    u, v, zl, D = u[valid], v[valid], zl[valid], D[valid]

    r0 = zl - D
    w = np.abs(r0) < gate
    for _ in range(3):  # MAD 重加权, 与 solve_ty_residual 同一策略
        mad = np.median(np.abs(r0[w] - np.median(r0[w]))) + 1e-6
        w = np.abs(r0) < max(0.03, 3.0 * mad)
        if w.sum() < 500:
            break
    r = r0[w]
    return dict(pose=os.path.basename(d.rstrip("/")), n=int(w.sum()),
                rms=float(np.sqrt((r**2).mean())),
                mean=float(r.mean()),  # ≈ -Δt.z 误差
                med=float(np.median(r)))


def main():
    dirs = sys.argv[1:] or sorted(glob.glob("turntable_output/20260923_*"))
    K, _ = load_camera_info("config/camera_info.yaml")
    print(f"{'pose':22s} {'n':>7s} | {'ours rms/mm':>11s} {'mean/mm':>8s} | {'matlab rms/mm':>13s} {'mean/mm':>8s}")
    agg = {"ours": [], "mlab": []}
    for d in dirs:
        ro = eval_frame(d, R_OURS, T_OURS, 90.0, K)
        rm = eval_frame(d, R_MLAB, T_MLAB, 0.0, K)
        if ro is None or rm is None:
            print(f"{os.path.basename(d):22s} 缺少 depth_cloud.ply/merged.ply, 跳过")
            continue
        agg["ours"].append(ro)
        agg["mlab"].append(rm)
        print(f"{ro['pose']:22s} {ro['n']:7d} | {ro['rms']*1e3:11.1f} {ro['mean']*1e3:8.1f} | "
              f"{rm['rms']*1e3:13.1f} {rm['mean']*1e3:8.1f}")
    for name, rows in agg.items():
        if not rows:
            continue
        n = sum(r_["n"] for r_ in rows)
        rms = float(np.sqrt(sum(r_["rms"]**2 * r_["n"] for r_ in rows) / n))
        mean = sum(r_["mean"] * r_["n"] for r_ in rows) / n
        med = float(np.median([r_["med"] for r_ in rows]))
        print(f"[{name}] 加权RMS={rms*1e3:.1f} mm  加权mean={mean*1e3:.1f} mm (≈-Δt.z误差)  逐帧median中位={med*1e3:.1f} mm  帧数={len(rows)}")


if __name__ == "__main__":
    main()
