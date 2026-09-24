#!/usr/bin/env python3
"""找板 v2: RANSAC 候选 -> 每候选 CC -> 矩形度排名 -> 滑窗 -> 回原始点云。

与 segment_board.py (extract_rect_plane) 的差异:
  旧: 全部候选滑窗, 按环带分选候选 (滑窗在每个候选的精修内点上做)
  新: 每个候选先做带内最大连通域, 用矩形度 (fill/长宽比/尺寸误差) 排名选出第一名,
      只对第一名滑窗 (旋转已知尺寸窗, 环带评分) —— debug 结论: 矩形度一眼定板,
      滑窗只需跑一次, 且只有旋转窗能正确排除板边缘外扩点。

输出 (帧目录下, 与 segment_board --save 同口径但不覆盖):
  board_rect_v2.ply          选中板点 (红)
  board_rect_overlay_v2.ply  全片云: 选中=红, 带内未选中=蓝, 其余=灰

用法: python3 segment_board_v2.py <帧目录或 merged.ply> [...] [--save]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from segment_board import (BAND_CELL, BOARD_L, BOARD_W, DIST_THR,
                           MIN_SEL, RANSAC_ITERS, RANSAC_K, RANSAC_MIN_IN,
                           REFINE_ITERS, REFINE_K, VERTICAL_NZ_MAX, VOXEL,
                           WIN_CELL, RING_CELLS, RING_PENALTY,
                           ANGLE_COARSE_STEP, ANGLE_FINE_RANGE, ANGLE_FINE_STEP,
                           _band_cc, _ransac_planes, _refine_plane,
                           _remove_ground, write_ply)


def _load_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                break
    P = np.loadtxt(path, skiprows=i)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    return P[:, :3]


def _rect_score(pts: np.ndarray, n: np.ndarray, c: np.ndarray) -> dict:
    """最大连通域矩形度: fill / 长宽比误差 / 尺寸误差 (对板 0.799x0.599)。"""
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    e1 = vt[0] - np.dot(vt[0], n) * n
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(n, e1)
    u, v = (pts - c) @ e1, (pts - c) @ e2
    uv = np.stack([u, v], axis=1).astype(np.float32)
    _, (rw, rh), _ = cv2.minAreaRect(uv)
    el, ew = max(rw, rh), min(rw, rh)
    iu = ((u - u.min()) / 0.01).astype(int)
    iv = ((v - v.min()) / 0.01).astype(int)
    occ = len(set(zip(iu.tolist(), iv.tolist()))) * 1e-4
    fill = occ / (el * ew) if el * ew > 0 else 0.0
    ar_err = abs(el / ew - BOARD_L / BOARD_W) / (BOARD_L / BOARD_W)
    se = abs(el - BOARD_L) / BOARD_L + abs(ew - BOARD_W) / BOARD_W
    return dict(score=fill / (1.0 + ar_err + se), fill=fill, ar=el / ew,
                se=se, el=el, ew=ew, e1=e1, e2=e2, u=u, v=v)


def _best_window(u, v, cell=WIN_CELL, ring_cells=RING_CELLS,
                 penalty=RING_PENALTY):
    """旋转已知尺寸滑窗: 评分 = 窗内 - penalty x 环带 (与 segment_board 同逻辑)。"""
    kw, kh = int(round(BOARD_L / cell)), int(round(BOARD_W / cell))
    R = ring_cells

    def search(angles):
        best = None
        for a in angles:
            r = np.radians(a)
            ca, sa = np.cos(r), np.sin(r)
            uu, vv = u * ca + v * sa, -u * sa + v * ca
            ulo, vlo = uu.min() - 0.05, vv.min() - 0.05
            nu = int((uu.max() - ulo) / cell) + 3
            nv = int((vv.max() - vlo) / cell) + 3
            G = np.zeros((nv + 2 * R, nu + 2 * R))
            iu = np.clip(((uu - ulo) / cell).astype(int) + R, 0, nu + 2 * R - 1)
            iv = np.clip(((vv - vlo) / cell).astype(int) + R, 0, nv + 2 * R - 1)
            np.add.at(G, (iv, iu), 1.0)
            I = np.pad(G.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            for pr in range(0, nv - kh + 1):
                for pc in range(0, nu - kw + 1):
                    win = (I[pr + R + kh, pc + R + kw] - I[pr + R, pc + R + kw]
                           - I[pr + R + kh, pc + R] + I[pr, pc])
                    big = (I[pr + 2 * R + kh, pc + 2 * R + kw]
                           - I[pr, pc + 2 * R + kw]
                           - I[pr + 2 * R + kh, pc] + I[pr, pc])
                    score = win - penalty * (big - win)
                    if best is None or score > best[0]:
                        best = (score, win, ulo + (pc + kw / 2) * cell,
                                vlo + (pr + kh / 2) * cell, a)
        return best

    coarse = search(np.arange(-90, 90.1, ANGLE_COARSE_STEP))
    return search(np.arange(coarse[4] - ANGLE_FINE_RANGE,
                            coarse[4] + ANGLE_FINE_RANGE + 0.1, ANGLE_FINE_STEP))


def extract_rect_plane_v2(P: np.ndarray, seed: int = 0, verbose: bool = True):
    """v2 链路: 去地面 -> 降采样 -> RANSAC 候选 -> 逐候选 (精修+CC+矩形度)
    -> 第一名 -> 滑窗 -> 回原始点云 (平面距离带 + 旋转矩形 SDF + 带内最大 CC)。"""
    def _log(msg: str):
        if verbose:
            print(msg)
    keep = _remove_ground(P)
    Pg = P[keep]
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pg)
    ds = np.asarray(pcd.voxel_down_sample(VOXEL).points)

    cands = []
    for rank, (nv, dd, idx) in enumerate(
            _ransac_planes(ds, DIST_THR, K=RANSAC_K, min_in=RANSAC_MIN_IN,
                           iters=RANSAC_ITERS, seed=seed)):
        n = nv / np.linalg.norm(nv)
        if abs(n[2]) > VERTICAL_NZ_MAX:
            continue                                     # 水平, 不可能是板
        n_r, d_r, pts = _refine_plane(ds[idx], n, dd,
                                      iters=REFINE_ITERS, k=REFINE_K)
        if d_r < 0:
            n_r, d_r = -n_r, -d_r
        c = pts.mean(axis=0)
        band = np.zeros(len(ds), bool)
        band[idx] = True
        cc = _band_cc(ds, _basis(pts, n_r, c), band, cell=BAND_CELL)
        cc_pts = ds[cc]
        if len(cc_pts) < 300:
            continue
        m = _rect_score(cc_pts, n_r, c)
        cands.append(dict(rank=rank, n=n_r, d=d_r, c=c, cc=cc,
                          cc_n=len(cc_pts), **m))
        _log(f"    [v2] plane_{rank:02d}: CC={len(cc_pts):>6}  "
             f"score={m['score']:.3f}  fill={m['fill']:.2f}  "
             f"ar={m['ar']:.2f}  外接={m['el'] * 100:.1f}x{m['ew'] * 100:.1f}cm")

    if not cands:
        return None, None, []
    cands.sort(key=lambda x: x["score"], reverse=True)
    best = cands[0]
    _log(f"  [v2] 矩形度第一名: plane_{best['rank']:02d}  "
         f"score={best['score']:.3f}  n=[{best['n'][0]:+.3f} "
         f"{best['n'][1]:+.3f} {best['n'][2]:+.3f}]  d={best['d']:+.3f}")

    # 滑窗只在第一名的 CC 上做
    res = _best_window(best["u"], best["v"])
    if res is None:
        return None, None, cands
    score, cnt, wc, hc, ang = res
    best.update(score=score, cnt=int(cnt), wc=wc, hc=hc, ang=ang)

    # 回原始点云 (§3.5 同口径): 平面距离带 + 旋转矩形 SDF + 带内最大 CC
    # 注意: wc/hc 是滑窗"旋转后坐标系"里的窗中心, 必须先旋转再减中心 (085036 踩雷记录
    # 见 segment_board.py 同款注释; 先减后转只在窗中心≈(u,v)原点或角度≈0 时碰巧正确)
    th = np.radians(ang)
    ca, sa = np.cos(th), np.sin(th)
    u = (Pg - best["c"]) @ best["e1"]
    v = (Pg - best["c"]) @ best["e2"]
    uu = u * ca + v * sa - wc
    vv = -u * sa + v * ca - hc
    band_g = np.abs(Pg @ best["n"] - best["d"]) < DIST_THR
    cc_g = _band_cc(Pg, best, band_g, cell=BAND_CELL)
    mask_g = band_g & cc_g \
        & (np.abs(uu) <= BOARD_L / 2 + VOXEL) & (np.abs(vv) <= BOARD_W / 2 + VOXEL)
    if mask_g.sum() < MIN_SEL:
        return None, None, cands
    mask = np.zeros(len(P), bool)
    mask[np.where(keep)[0]] = mask_g
    best["n_sel"] = int(mask_g.sum())
    best["band_cc_n"] = int(cc_g.sum())
    return mask, best, cands


def _basis(pts, n, c):
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    e1 = vt[0] - np.dot(vt[0], n) * n
    e1 /= np.linalg.norm(e1) + 1e-12
    return {"c": c, "e1": e1, "e2": np.cross(n, e1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="找板 v2: CC + 矩形度选候选")
    ap.add_argument("paths", nargs="+", help="帧目录或 merged.ply")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    print(f"{'帧':<12}{'窗内点':>8}{'实测尺寸cm':>14}{'环带分':>9}{'选中':>8}{'原占比':>7}")
    for p in a.paths:
        d = p if os.path.isdir(p) else os.path.dirname(p)
        mp = os.path.join(d, "merged.ply")
        if not os.path.exists(mp):
            print(f"  {os.path.basename(d):<10} 缺 merged.ply, 跳过")
            continue
        P = _load_ply_xyz(mp)
        mask, info, _ = extract_rect_plane_v2(P)
        name = os.path.basename(d)
        if mask is None:
            print(f"  {name:<10} FAIL")
            continue
        sel = P[mask]
        c0 = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - c0, full_matrices=False)
        uv = np.stack([(sel - c0) @ vt[0], (sel - c0) @ vt[1]],
                      axis=1).astype(np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(uv)
        el, ew = max(rw, rh) * 100, min(rw, rh) * 100
        ratio = info["n_sel"] / max(info["band_cc_n"], 1)
        print(f"  {name:<10}{info['cnt']:>8}{el:>6.1f}x{ew:>5.1f}"
              f"{info['score']:>9.0f}{info['n_sel']:>8}{ratio:>7.2f}")
        if a.save:
            write_ply(os.path.join(d, "board_rect_v2.ply"), sel,
                      np.tile([255, 30, 30], (len(sel), 1)).astype(np.uint8))
            cols = np.full((len(P), 3), 120, np.uint8)
            cols[mask] = (255, 30, 30)
            write_ply(os.path.join(d, "board_rect_overlay_v2.ply"), P, cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
