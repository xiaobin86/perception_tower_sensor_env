import glob
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, "/workspace")
import segment_board as sb


def size2d(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    uv = np.stack([(pts - c) @ vt[0], (pts - c) @ vt[1]], axis=1).astype(np.float32)
    _, (rw, rh), _ = cv2.minAreaRect(uv)
    return max(rw, rh) * 100, min(rw, rh) * 100


print(f"{'帧':<11}{'平面点':>7}{'平面尺寸cm':>13}{'裁剪点':>7}{'裁剪尺寸cm':>13}{'占比':>7}"
      f"{'原始带点':>8}{'原始选中':>8}{'原占比':>7}")
for d in sorted(glob.glob("/workspace/turntable_output/20260918_0*")):
    name = os.path.basename(d)
    mp = os.path.join(d, "merged.ply")
    if not os.path.exists(mp):
        continue
    with open(mp) as f:
        for i, l in enumerate(f, 1):
            if l.strip() == "end_header":
                break
    P = np.loadtxt(mp, skiprows=i)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    ds = np.asarray(pcd.voxel_down_sample(0.01).points)
    mask, info = sb.extract_rect_plane(P)
    if mask is None:
        print(f"  {name:<9} FAIL")
        continue
    # 选中候选的平面: 在 ds 上按平面模型取内点
    n, dd = info["n"], info["d"]
    band_ds = np.abs(ds @ n - dd) < 0.02
    band_P = np.abs(P @ n - dd) < 0.02
    pl_cnt = int(band_ds.sum())
    pl_l, pl_w = size2d(ds[band_ds])
    sel = P[mask]
    se_l, se_w = size2d(sel)
    ratio_ds = info["cnt"] / max(pl_cnt, 1)
    ratio_P = int(mask.sum()) / max(int(band_P.sum()), 1)
    print(f"  {name:<9}{pl_cnt:>7}{pl_l:>6.0f}x{pl_w:>5.0f}{info['cnt']:>7}"
          f"{se_l:>6.0f}x{se_w:>5.0f}{ratio_ds:>7.2f}"
          f"{int(band_P.sum()):>8}{int(mask.sum()):>8}{ratio_P:>7.2f}")
