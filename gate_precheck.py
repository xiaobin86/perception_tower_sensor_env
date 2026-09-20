#!/usr/bin/env python3
"""标定前门禁预审: 对候选帧逐帧跑 segment_board + 质量门(与 calibrate_camera_lidar 同一口径),
打印每帧 尺寸/尺寸误差/占比 与 采纳/拒绝 判定, 不重算外参。

用法: python3 gate_precheck.py <帧目录...>
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_camera_lidar import read_ply_xyz
from segment_board import extract_rect_plane

RATIO_MIN = 0.85
SIZE_MIN_L, SIZE_MIN_S, SIZE_ERR_MAX = 0.78, 0.54, 0.16


def main() -> int:
    dirs = sys.argv[1:]
    if not dirs:
        print("用法: python3 gate_precheck.py <帧目录...>")
        return 1
    n_ok = 0
    print(f"{'帧':<18}{'尺寸cm':>14}{'尺寸误差':>9}{'占比':>7}  门禁结果")
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        mp = os.path.join(d, "merged.ply")
        if not os.path.exists(mp):
            print(f"{name:<18}{'--':>14}{'--':>9}{'--':>7}  ✗ 拒绝 (无 merged.ply)")
            continue
        P = read_ply_xyz(mp)
        mask, info = extract_rect_plane(P)
        if mask is None:
            print(f"{name:<18}{'--':>14}{'--':>9}{'--':>7}  ✗ 拒绝 (无合格候选平面)")
            continue
        sel = P[mask]
        c0 = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - c0, full_matrices=False)
        uv = np.stack([(sel - c0) @ vt[0], (sel - c0) @ vt[1]], axis=1).astype(np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(uv)
        lo, sh = max(rw, rh), min(rw, rh)
        se = abs(lo - 0.84) / 0.84 + abs(sh - 0.60) / 0.60
        ratio = info["n_sel"] / max(int(info.get("band_cc_n", 1)), 1)
        ok = lo >= SIZE_MIN_L and sh >= SIZE_MIN_S and se <= SIZE_ERR_MAX and ratio >= RATIO_MIN
        if ok:
            n_ok += 1
            verdict = "✓ 采纳"
        else:
            why = []
            if ratio < RATIO_MIN:
                why.append(f"占比{ratio:.2f}<{RATIO_MIN}")
            if lo < SIZE_MIN_L or sh < SIZE_MIN_S:
                why.append("尺寸过小")
            if se > SIZE_ERR_MAX:
                why.append(f"尺寸误差{se*100:.0f}%>16%")
            verdict = "✗ 拒绝 (" + ", ".join(why) + ")"
        print(f"{name:<18}{lo * 100:>6.1f}x{sh * 100:>5.1f}{se * 100:>8.1f}%"
              f"{ratio:>7.2f}  {verdict}")
    print(f"\n采纳 {n_ok}/{len(dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
