#!/usr/bin/env python3
"""从点云里 ROI 出"已知尺寸矩形板"（0.60×0.84 m），姿态任意。

思路（不依赖相机外参）:
  1) 找板所在平面（RANSAC 最大竖直平面）→ 得平面法向与平面内基 e1/e2
  2) 把板面点投到平面内得 2D 点集 → 栅格化成占用图 → 取占用边界点
  3) 对边界点做 RANSAC 直线拟合，依次取最长若干条（板上碎块只会给短线段，会被丢掉）
  4) 把最长四条按方向分成两组（两组近似互相垂直）→ 求四角
  5) 用已知尺寸 (0.60×0.84) 约束精修：中心=四角均值，轴=两组方向正交化，尺寸偏差同时报出

输出: 矩形中心(三维)/轴向/实测边长 + 板面点云 board_roi.ply + 调试图 *_roi.png
用法: python3 roi_board.py <merged.ply 或 扫描目录> [--save]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_board as eb


def ransac_line(P: np.ndarray, tol: float, iters: int = 1500, seed: int = 0):
    """在 2D 点集里找一条直线（能覆盖最多点），返回 (点法向, 偏移, 内点掩码)。"""
    rng = np.random.default_rng(seed)
    n = len(P)
    if n < 20:
        return None
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = P[j] - P[i]
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        t = np.array([-d[1], d[0]]) / L
        off = float(t @ P[i])
        cnt = int((np.abs(P @ t - off) < tol).sum())
        if best is None or cnt > best[0]:
            best = (cnt, t, off)
    if best is None:
        return None
    cnt, t, off = best
    return t, off, (np.abs(P @ t - off) < tol)


def _slide_fallback(pts, r, patch, n, e1, e2, c0, u, v, W, H, cell, why):
    """线拟合不合格时的兜底: 平面内已知尺寸滑窗 + 面内角搜索（不用外参）。"""
    rel = patch - c0
    best = (-1, 0.0, 0.0, 0.0)
    for a in np.arange(-25, 25.1, 2.0):
        rr = np.radians(a); ca, sa = np.cos(rr), np.sin(rr)
        a1 = ca*np.array([1.0, 0]) + sa*np.array([0, 1.0])
        a2 = -sa*np.array([1.0, 0]) + ca*np.array([0, 1.0])
        uu, vv = u, v
        ulo, vlo = uu.min()-0.05, vv.min()-0.05
        nu = int((uu.max()-ulo)/0.01)+4; nv = int((vv.max()-vlo)/0.01)+4
        G = np.zeros((nv, nu))
        iu = np.clip(((uu-ulo)/0.01).astype(int), 0, nu-1); iv = np.clip(((vv-vlo)/0.01).astype(int), 0, nv-1)
        np.add.at(G, (iv, iu), 1.0)
        I = np.pad(G.cumsum(0).cumsum(1), ((1,0),(1,0)))
        kw, kh = int(round(W/0.01)), int(round(H/0.01))
        bs, br, bc = -1, 0, 0
        for i2 in range(0, nv-kh):
            row = I[i2+kh, kw:] - I[i2, kw:] - I[i2+kh, :-kw] + I[i2, :-kw]
            j = int(np.argmax(row))
            if row[j] > bs: bs, br, bc = float(row[j]), i2, j
        if bs > best[0]:
            best = (bs, a, ulo+(bc+kw/2)*0.01, vlo+(br+kh/2)*0.01)
    cnt, ang, uc, vc = best
    rr = np.radians(ang)
    d1 = np.cos(rr)*np.array([1.0,0]) + np.sin(rr)*np.array([0,1.0])
    d2 = -np.sin(rr)*np.array([1.0,0]) + np.cos(rr)*np.array([0,1.0])
    ctr2 = np.array([uc, vc])
    ctr3 = c0 + ctr2[0]*e1 + ctr2[1]*e2
    uv_rel = np.column_stack([u, v]) - ctr2[None, :]
    inside = (np.abs(uv_rel @ d1) <= H/2) & (np.abs(uv_rel @ d2) <= W/2)
    ideal = np.array([ctr2 + d1*(s1*H/2) + d2*(s2*W/2) for s1 in (-1,1) for s2 in (-1,1)])
    return {"normal": n, "center": ctr3, "center2": ctr2, "e1": e1, "e2": e2,
            "d1": d1, "d2": d2, "meas_W": W, "meas_H": H, "n_lines": 0,
            "inlier3": patch[inside], "patch": patch, "lines": [], "corners": ideal,
            "ideal": ideal, "score": float("nan"), "method": f"滑窗({why})"}


def roi_rectangle(pts: np.ndarray, W: float = 0.60, H: float = 0.84,
                  cell: float = 0.012, n_lines: int = 10):
    r = eb.find_board(pts)
    if r is None:
        raise SystemExit("未找到板平面")
    n = np.asarray(r["n"], float)
    patch = eb.board_points(pts, r)
    e1, e2 = eb.plane_basis(n)
    c0 = patch.mean(axis=0)
    rel = patch - c0
    u, v = rel @ e1, rel @ e2

    # ---- 占用栅格 + 边界点 ----
    lo = np.array([u.min(), v.min()]) - cell
    idx = np.floor((np.column_stack([u, v]) - lo) / cell).astype(int)
    shape = idx.max(0) + 2
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1]] = True
    er = cv2.erode(occ.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    bd = occ & ~er
    is_bd = bd[idx[:, 0], idx[:, 1]]
    Pb = np.column_stack([u, v])[is_bd]

    # ---- RANSAC 依次取最长直线 ----
    tol = 1.6 * cell
    lines, P = [], Pb.copy()
    for k in range(n_lines):
        res = ransac_line(P, tol, seed=k)
        if res is None:
            break
        t, off, m = res
        if m.sum() < 40:
            break
        q = P[m]
        along = q @ np.array([-t[1], t[0]])
        length = float(along.max() - along.min())
        lines.append({"t": t, "off": off, "len": length,
                      "p0": P[m][np.argmin(along)], "p1": P[m][np.argmax(along)],
                      "n": int(m.sum()), "mid": q.mean(axis=0)})
        P = P[~m]
    if len(lines) < 4:
        raise SystemExit(f"只找到 {len(lines)} 条边（需要 4 条）")

    lines.sort(key=lambda L: -L["len"])
    lines = lines[:4]

    # ---- 主轴法：以最长边为主轴, 找对边, 用已知尺寸打分 ----
    def line_dir(L):
        t = L["t"]                       # 法向
        return np.array([-t[1], t[0]])   # 方向

    cand = []
    for pi, Lp in enumerate(lines):
        d1 = line_dir(Lp); d1 /= np.linalg.norm(d1) + 1e-12
        par, perp = [], []
        for k, L in enumerate(lines):
            if k == pi:
                continue
            d = abs(float(line_dir(L) @ d1))
            if d > 0.75:
                par.append(k)
            elif d < 0.45:
                perp.append(k)
        if not par or not perp:
            continue
        # 对边：与主轴平行的那些里, 与主轴距离最接近 H 或 W 的
        best_par, best_sep = None, None
        for k in par:
            # 两条平行线的间距 = 在 d1 法向上偏移差
            sep = abs(float((Lp["mid"] - lines[k]["mid"]) @ Lp["t"]))
            for target in (H, W):
                err = abs(sep - target)
                if best_sep is None or err < best_sep[0]:
                    best_sep = (err, k, sep, target)
        if best_sep is None:
            continue
        _, kpar, sep1, tgt1 = best_sep
        # 另一轴的间距（剩下两条垂直边之间）
        if len(perp) < 2:
            continue
        d2 = line_dir(lines[perp[0]]); d2 /= np.linalg.norm(d2) + 1e-12
        n2 = np.array([-d2[1], d2[0]])
        sep2 = abs(float((lines[perp[0]]["mid"] - lines[perp[1]]["mid"]) @ n2))
        tgt2 = H if tgt1 == W else W
        score = abs(sep1 - tgt1) + abs(sep2 - tgt2)
        cand.append({"pi": pi, "par": kpar, "perp": perp[:2], "d1": d1, "d2": d2,
                     "sep1": sep1, "sep2": sep2, "score": score,
                     "H": max(sep1, sep2), "W": min(sep1, sep2)})
    if not cand or min(c["score"] for c in cand) > 0.25:
        return _slide_fallback(pts, r, patch, n, e1, e2, c0, u, v, W, H, cell,
                               "边组合不合格" if not cand else
                               f"残差过大 {min(c['score'] for c in cand)*100:.1f}cm")
    best_c = min(cand, key=lambda c: c["score"])
    if best_c["score"] > 0.25:
        raise SystemExit(f"矩形拟合不合格: 主轴法残差 {best_c['score']*100:.1f} cm "
                         f"(实测 {best_c['H']*100:.1f}×{best_c['W']*100:.1f} cm, 应 84×60)")

    # 四角 = 主轴/对边 与 两条垂直边 的交点
    Ls = [lines[best_c["pi"]], lines[best_c["par"]]] + [lines[k] for k in best_c["perp"]]

    def inter(L1, L2):
        A = np.array([L1["t"], L2["t"]]); b = np.array([L1["off"], L2["off"]])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, b)

    corners = []
    for i in (0, 1):
        for j in (2, 3):
            q = inter(Ls[i], Ls[j])
            if q is not None:
                corners.append(q)
    if len(corners) < 4:
        raise SystemExit("四角求解失败")
    corners = np.array(corners[:4])
    ctr2 = corners.mean(axis=0)
    d1, d2 = best_c["d1"], best_c["d2"]
    d2 = d2 - (d2 @ d1) * d1; d2 /= np.linalg.norm(d2) + 1e-12
    meas_H, meas_W = best_c["H"], best_c["W"]
    ctr3 = c0 + ctr2[0] * e1 + ctr2[1] * e2
    ideal = np.array([ctr2 + d1 * (sH * meas_H / 2) + d2 * (sW * meas_W / 2)
                      for sH in (-1, 1) for sW in (-1, 1)])
    uv_rel = np.column_stack([u, v]) - ctr2[None, :]
    a1 = uv_rel @ d1
    a2 = uv_rel @ d2
    inside = ((np.abs(a1) <= meas_H / 2 + 0.02) & (np.abs(a2) <= meas_W / 2 + 0.02))
    inlier3 = patch[inside]
    return {"normal": n, "center": ctr3, "center2": ctr2, "e1": e1, "e2": e2,
            "d1": d1, "d2": d2, "meas_W": meas_W, "meas_H": meas_H,
            "n_lines": len(lines), "inlier3": inlier3, "patch": patch,
            "lines": lines, "corners": corners, "ideal": ideal, "score": best_c["score"],
            "method": "边缘直线法"}


def main() -> int:
    ap = argparse.ArgumentParser(description="从点云 ROI 已知尺寸矩形板 (0.60×0.84)")
    ap.add_argument("path")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    path = a.path if a.path.endswith(".ply") else os.path.join(a.path, "merged.ply")
    pts = eb.load_ply(path)
    R = roi_rectangle(pts)
    print(f"平面法向 n = {np.round(R['normal'],4).tolist()}")
    print(f"板中心(三维) = {np.round(R['center'],4).tolist()}")
    print(f"边长实测: 长边 {R['meas_H']*100:.1f} cm (应 84)   短边 {R['meas_W']*100:.1f} cm (应 60)")
    print(f"尺寸偏差: 长边 {100*(R['meas_H']/0.84-1):+.1f}%   短边 {100*(R['meas_W']/0.60-1):+.1f}%")
    print(f"用到的边: {R['n_lines']} 条, 各边点数 {[L['n'] for L in R['lines']]}")
    print(f"板面点(理想矩形内): {len(R['inlier3'])} / {len(R['patch'])}")
    if a.save:
        outdir = os.path.dirname(path)
        with open(os.path.join(outdir, "board_roi.ply"), "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(R['inlier3'])}\n"
                    "property float x\nproperty float y\nproperty float z\nend_header\n")
            np.savetxt(f, R["inlier3"], fmt="%.6f")
        print(f"已写 {os.path.join(outdir, 'board_roi.ply')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
