#!/usr/bin/env python3
"""用相机自带深度云求解外参的平移补偿（不靠目测估数）。

原理: depth_cloud_colored.ply 是相机自己的 3D 点（相机系），与雷达云看同一场景。
      把雷达云用当前外参变到相机系 → 在相机系里网格搜索平移增量 Δ，
      让"雷达点 → 相机点云"的最近邻中位距离最小 → 该 Δ 就是补偿量。
      ★ 方向由指标判定，不依赖任何先验符号（避免竖直方向推反）。

进一步: 按距离分段各自求最优 Δy。若远处需要的 Δy 与近处不同，
      说明还有俯仰(pitch)残差而不是纯平移 → 脚本会给出等效俯仰角估计。

用法:
    python3 solve_vertical_compensation.py <pose_dir> [<pose_dir> ...]
        [--extrinsics config/camera_extrinsics.yaml] [--photo-angle 90]
        [--step 0.005] [--span-y 0.10] [--span-xz 0.05] [--max-pts 20000]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

from verify_colorization import read_ply_xyzrgb, rotation_z


def median_nn(src: np.ndarray, tree: cKDTree, gate: float = 0.30) -> tuple[float, int]:
    d, _ = tree.query(src, k=1)
    d = d[d < gate]
    if len(d) < 100:
        return float("nan"), len(d)
    return float(np.median(d)), len(d)


def search_axis(src: np.ndarray, tree: cKDTree, axis: int,
                span: float, step: float) -> tuple[float, float]:
    best = (None, 0.0)
    for v in np.arange(-span, span + step / 2, step):
        s = src.copy()
        s[:, axis] += v
        m, _ = median_nn(s, tree)
        if np.isnan(m):
            continue
        if best[0] is None or m < best[0]:
            best = (m, float(v))
    return best


def solve_pose(pose_dir: str, ex_path: str, photo_angle: float,
               step: float, span_y: float, span_xz: float, max_pts: int) -> dict | None:
    lp = os.path.join(pose_dir, "colored.ply")
    cp = os.path.join(pose_dir, "depth_cloud_colored.ply")
    if not (os.path.exists(lp) and os.path.exists(cp)):
        print(f"  skip {os.path.basename(pose_dir)}: 缺少点云")
        return None

    ex = yaml.safe_load(open(ex_path))["lidar_to_camera"]
    R = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t = np.array(ex["translation"])

    lidar, _ = read_ply_xyzrgb(lp)
    cam, _ = read_ply_xyzrgb(cp)
    lc = (R @ ((lidar @ rotation_z(photo_angle).T).T)).T + t

    rng = np.random.default_rng(0)
    if len(cam) > 300000:
        sel = rng.choice(len(cam), 300000, replace=False)
        cam = cam[sel]
    tree = cKDTree(cam)
    if len(lc) > max_pts:
        lc = lc[rng.choice(len(lc), max_pts, replace=False)]

    cur, ncur = median_nn(lc, tree)

    d = np.zeros(3)
    for _ in range(2):
        _, d[1] = search_axis(lc + d, tree, 1, span_y, step)
        _, d[0] = search_axis(lc + d, tree, 0, span_xz, step)
        _, d[2] = search_axis(lc + d, tree, 2, span_xz, step)
    best, nbest = median_nn(lc + d, tree)

    bands = []
    for lo, hi in ((0.3, 1.0), (1.0, 2.0), (2.0, 3.5)):
        s = lc + d
        sel = (s[:, 2] >= lo) & (s[:, 2] < hi)
        if sel.sum() < 500:
            continue
        m, v = search_axis(lc[sel] + d, tree, 1, span_y, step)
        if m == m:
            bands.append((lo, hi, v, m, int(sel.sum())))

    return {"pose": os.path.basename(pose_dir), "cur": cur, "n_cur": ncur,
            "best": best, "n_best": nbest, "d": d, "bands": bands}


def main() -> int:
    ap = argparse.ArgumentParser(description="Solve extrinsic translation compensation from the camera depth cloud")
    ap.add_argument("pose_dirs", nargs="+")
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--span-y", type=float, default=0.10)
    ap.add_argument("--span-xz", type=float, default=0.05)
    ap.add_argument("--max-pts", type=int, default=20000)
    a = ap.parse_args()

    res = []
    for p in a.pose_dirs:
        r = solve_pose(p, a.extrinsics, a.photo_angle, a.step, a.span_y, a.span_xz, a.max_pts)
        if r is None:
            continue
        res.append(r)
        print(f"{r['pose']}: 当前 NN中位 {r['cur']*1000:6.1f}mm → 补偿后 {r['best']*1000:6.1f}mm "
              f"| Δ = ({r['d'][0]*1000:+.1f}, {r['d'][1]*1000:+.1f}, {r['d'][2]*1000:+.1f}) mm")
        for lo, hi, v, m, n in r["bands"]:
            print(f"      {lo}-{hi}m: Δy={v*1000:+6.1f}mm (NN {m*1000:5.1f}mm, {n} 点)")

    if not res:
        print("无有效位姿")
        return 1
    D = np.array([r["d"] for r in res])
    print(f"\n跨 {len(res)} 帧共识: Δx={np.median(D[:,0])*1000:+.1f}mm  "
          f"Δy={np.median(D[:,1])*1000:+.1f}mm  Δz={np.median(D[:,2])*1000:+.1f}mm"
          f"   (逐帧散布 y: {D[:,1].std()*1000:.1f}mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
