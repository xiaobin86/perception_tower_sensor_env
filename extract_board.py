#!/usr/bin/env python3
"""直接从 merged.ply 里找棋盘格板：只找平面 + 取形状接近 0.60×0.84 m 的那块。

按用户要求：不加任何下限/高度/范围/先验限制。
流程: RANSAC 找平面 → 平面内 2D 密度图 → 最大连通致密块 → minAreaRect 贴合
      → 尺寸落在 0.60×0.84 的 ±20% 内才接受。

输出: 每帧的 (尺寸, 点数) 打印；并把板面点写成 <dir>/board_selected.ply 供目测。
用法: python3 extract_board.py turntable_output/<dir> [...]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from scipy import ndimage


def load_ply(path: str) -> np.ndarray:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                break
    return np.loadtxt(path, skiprows=i)[:, :3]


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    v = up - (up @ n) * n
    v /= np.linalg.norm(v)
    return np.cross(n, v), v


def dense_core_mask(uv: np.ndarray, cell: float = 0.04,
                    frac: float = 0.15) -> np.ndarray:
    lo = uv.min(axis=0) - cell
    idx = np.floor((uv - lo) / cell).astype(int)
    shape = idx.max(axis=0) + 2
    grid = np.zeros(shape, dtype=np.int32)
    np.add.at(grid, (idx[:, 0], idx[:, 1]), 1)
    dense = grid >= max(3, frac * grid.max())
    lab, n = ndimage.label(dense)
    if n == 0:
        return np.zeros(len(uv), bool)
    sizes = ndimage.sum(dense, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return lab[idx[:, 0], idx[:, 1]] == best


def find_board(pts: np.ndarray, width: float = 0.60, height: float = 0.84,
               tol: float = 0.20, band: float = 0.02, rounds: int = 8):
    rng = np.random.default_rng(0)
    P = pts[rng.choice(len(pts), min(60000, len(pts)), replace=False)]
    for _ in range(rounds):
        if len(P) < 1500:
            return None
        best = None
        for _ in range(2000):
            i, j, k = rng.choice(len(P), 3, replace=False)
            n = np.cross(P[j] - P[i], P[k] - P[i])
            L = np.linalg.norm(n)
            if L < 1e-9:
                continue
            n /= L
            d = float(n @ P[i])
            cnt = int((np.abs(P @ n - d) < band).sum())
            if best is None or cnt > best[0]:
                best = (cnt, n, d)
        if best is None:
            return None
        cnt, n, d = best
        inl_mask = np.abs(P @ n - d) < band
        inl = P[inl_mask]
        e1, e2 = plane_basis(n)
        c = inl.mean(axis=0)
        rel = inl - c
        uv = np.column_stack([rel @ e1, rel @ e2])
        core = dense_core_mask(uv)
        if core.sum() < 800:
            P = P[~inl_mask]
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(uv[core].astype(np.float32))
        long_s, short_s = max(w, h), min(w, h)
        if abs(long_s / height - 1) <= tol and abs(short_s / width - 1) <= tol:
            return {"n": n, "d": d, "center": c, "e1": e1, "e2": e2,
                    "size": (long_s, short_s), "core": core, "inl": inl, "uv": uv, "ang": ang}
        P = P[~inl_mask]
    return None


def board_points(points: np.ndarray, r: dict, margin: float = 0.02) -> np.ndarray:
    """用 find_board 得到的平面+致密核矩形，在任意点云上取出该板的点（全密度）。"""
    n, d0, c, e1, e2 = r["n"], r["d"], r["center"], r["e1"], r["e2"]
    (cx, cy), (w, h), ang = cv2.minAreaRect(r["uv"][r["core"]].astype(np.float32))
    band = np.abs(points @ n - d0) < 0.02
    uv = np.column_stack([(points[band] - c) @ e1, (points[band] - c) @ e2])
    th = np.radians(ang); ca, sa = np.cos(th), np.sin(th)
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    ru, rv = du * ca + dv * sa, -du * sa + dv * ca
    rlong, rshort = (rv, ru) if h >= w else (ru, rv)
    keep = ((np.abs(rshort) < min(w, h) / 2 + margin)
            & (np.abs(rlong) < max(w, h) / 2 + margin))
    return points[band][keep]


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract the checkerboard plane from merged.ply")
    ap.add_argument("pose_dirs", nargs="+")
    ap.add_argument("--save", action="store_true", help="把板面点写成 <dir>/board_selected.ply")
    a = ap.parse_args()

    for d in a.pose_dirs:
        path = os.path.join(d, "merged.ply")
        if not os.path.exists(path):
            print(f"skip {d}: 无 merged.ply")
            continue
        pts = load_ply(path)
        r = find_board(pts)
        if r is None:
            print(f"{os.path.basename(d)}: ✗ 未找到尺寸接近 0.60×0.84 的平面块")
            continue
        print(f"{os.path.basename(d)}: ✓ 尺寸 {r['size'][0]:.2f}×{r['size'][1]:.2f} m "
              f"(期望 0.84×0.60)  平面内点 {len(r['inl'])}  致密核 {int(r['core'].sum())}")
        if a.save:
            sel = r["inl"][r["core"]]
            out = os.path.join(d, "board_selected.ply")
            with open(out, "w") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(sel)}\n"
                        "property float x\nproperty float y\nproperty float z\nend_header\n")
                np.savetxt(f, sel, fmt="%.6f")
            print(f"    → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
