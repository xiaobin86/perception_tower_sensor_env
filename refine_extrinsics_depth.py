#!/usr/bin/env python3
"""用深度点云联合微调外参的小旋转与平移（绕激光竖直轴偏航 + 相机系竖直平移）。

代价 = 多帧"变换后激光点到深度云的最近邻中位距离"的平均（对单帧离群稳健）。
参数在拍照系激光坐标下定义：R_new = R · Rz(rz)Ry(ry)Rx(rx)，t_new = t + (tx,ty,tz)_camera。

用法:
    python3 refine_extrinsics_depth.py --params ty,rz                 # 仅扫描并打印
    python3 refine_extrinsics_depth.py --params ty,rz --write         # 写回（自动备份 .bak）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def load_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path}: no end_header")
    return np.loadtxt(path, skiprows=i, dtype=np.float64)[:, :3]


def rot_axis(axis: str, deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine extrinsics against the depth cloud")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--data-root", default="turntable_output")
    parser.add_argument("--scans", type=int, default=6)
    parser.add_argument("--params", default="ty,rz", help="逗号分隔，可选 tx,ty,tz,rx,ry,rz")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    big = {"tx": 0.2, "ty": 0.2, "tz": 0.2, "rx": 3.0, "ry": 3.0, "rz": 3.0}
    scale = {"tx": 1.0, "ty": 1.0, "tz": 1.0, "rx": 1.0, "ry": 1.0, "rz": 1.0}

    K = np.array(yaml.safe_load(open(args.camera_info))["k"]).reshape(3, 3)
    dist = np.array(yaml.safe_load(open(args.camera_info))["d"]).reshape(-1, 1)
    ex = yaml.safe_load(open(args.extrinsics))["lidar_to_camera"]
    R0 = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t0 = np.array(ex["translation"])
    Rz = rot_axis("z", args.photo_angle)
    rvec0 = cv2.Rodrigues(R0 @ Rz)[0]

    dirs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.exists(os.path.join(args.data_root, d, "merged.ply"))
                  and os.path.exists(os.path.join(args.data_root, d, "depth_cloud.ply")))
    step = max(1, len(dirs) // args.scans)
    dirs = dirs[::step][:args.scans]

    cloud, trees, zc = [], [], []
    for name in dirs:
        p = os.path.join(args.data_root, name)
        merged = load_ply_xyz(os.path.join(p, "merged.ply"))
        depth = load_ply_xyz(os.path.join(p, "depth_cloud.ply"))
        depth = depth[np.linalg.norm(depth, axis=1) < 8.0]
        if len(depth) < 1000:
            continue
        img = cv2.imread(os.path.join(p, "color.png"))
        h, w = img.shape[:2]
        photo = merged @ Rz.T
        pc = (R0 @ photo.T).T + t0
        proj, _ = cv2.projectPoints(merged, rvec0, t0, K, dist)
        proj = proj.reshape(-1, 2)
        keep = ((pc[:, 2] > 0.2) & np.isfinite(proj).all(axis=1)
                & (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
                & (np.linalg.norm(merged, axis=1) < 8.0))
        idx = np.where(keep)[0]
        if len(idx) < 1500:
            continue
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, min(args.samples, len(idx)), replace=False)
        cloud.append(photo[idx].copy())
        zc.append((R0 @ photo[idx].T).T[:, 2] + t0[2])
        trees.append(cKDTree(depth))
        print(f"  统计帧 {name}: {len(idx)} 点")

    if not cloud:
        print("no usable scan", file=sys.stderr)
        return 1

    def cost(values, detail=False):
        update = {k: 0.0 for k in big}
        update.update(dict(zip(params, values)))
        dR = rot_axis("z", update["rz"]) @ rot_axis("y", update["ry"]) @ rot_axis("x", update["rx"])
        R = R0 @ dR
        t = t0 + np.array([update["tx"], update["ty"], update["tz"]])
        meds = []
        for pts, tree, z in zip(cloud, trees, zc):
            pc = (R @ pts.T).T + t
            d, _ = tree.query(pc, k=1)
            meds.append(np.median(d))
        meds = np.array(meds)
        if detail:
            return meds
        return float(np.mean(meds))

    base = cost([0.0] * len(params), detail=True)
    print(f"\n初始: 各帧中位NN(mm) = {np.round(base*1000,1).tolist()}, 平均={base.mean()*1000:.1f}mm")

    bounds = [(-big[p], big[p]) for p in params]
    res = minimize(cost, np.zeros(len(params)), method="Powell", bounds=bounds,
                   options={"xtol": 1e-4, "ftol": 1e-8, "maxiter": 60})
    vals = res.x
    print(f"\n最优修正: " + ", ".join(f"{p}={v*scale.get(p,1)*1000 if p.startswith('t') else v:.3f}" +
                                     ("mm" if p.startswith("t") else "°") for p, v in zip(params, vals)))
    after = cost(vals, detail=True)
    print(f"各帧中位NN(mm) = {np.round(after*1000,1).tolist()}, 平均={after.mean()*1000:.1f}mm")

    dR = rot_axis("z", dict(zip(params, vals)).get("rz", 0.0)) @ \
         rot_axis("y", dict(zip(params, vals)).get("ry", 0.0)) @ \
         rot_axis("x", dict(zip(params, vals)).get("rx", 0.0))
    R_new = R0 @ dR
    t_new = t0 + np.array([dict(zip(params, vals)).get(k, 0.0) for k in ("tx", "ty", "tz")])

    if args.write:
        shutil.copy(args.extrinsics, args.extrinsics + ".bak")
        data = {"lidar_to_camera": {"rotation_matrix": R_new.flatten().tolist(),
                                    "translation": t_new.tolist()}}
        with open(args.extrinsics, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"\n已写回 {args.extrinsics}（备份 {args.extrinsics}.bak）")
    else:
        print("\n（未写文件；加 --write 应用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
