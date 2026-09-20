#!/usr/bin/env python3
"""按 rect_plane_segmentation_guide.md 的管线, 从点云中分割已知尺寸矩形板 (0.84x0.60 m)。

管线(对应文档章节):
  3.1 体素降采样(最终分割回原始点云)
  3.2 多候选平面: 确定性多种子 RANSAC, 每轮取最大平面并剔除内点, 取前 K 个
      (不取最大平面 -> 不被墙/地吸走; 跳过 |n_z|>0.5 的非竖直平面)
  3.3 候选内点 -> 平面内 2D 投影 -> 滑已知尺寸窗
  3.4 评分: 窗内点 - 3 x 边缘环带点(环带宽 ~3cm)
      利用"板是孤立的"先验: 对齐板边的窗外环带几乎为空, 偏窗/墙窗环带密集
  3.5 最优模型回原始点云做包含测试(平面距离 + 旋转矩形 SDF, expand=voxel)

用法: python3 segment_board.py <目录或 merged.ply> [...] [--save]
输出: 每帧 board_rect.ply(仅板点) + board_rect_overlay.ply(全片云, 板=红)
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import open3d as o3d

# ============================== 可调配置 ==============================
# 标定板
BOARD_L = 0.84                    # 板外形长边 (m)
BOARD_W = 0.60                    # 板外形短边 (m)
# 点云预处理
VOXEL = 0.01                      # 体素降采样边长: 只用于候选生成加速, 最终分割回原始点云
GROUND_THR = 0.02                 # 地面剔除的平面距离容差 ±2cm
GROUND_ITERS = 1500               # 地面 RANSAC 迭代数
GROUND_NZ_MIN = 0.95              # 地面法向 |n_z| 下限(只接受水平面)
# RANSAC 候选平面
DIST_THR = 0.02                   # 平面距离容差 ±2cm(RANSAC内点/最终带共用)。
                                  # 实测 ±1cm 会丢斜射帧(072600/072638 FAIL), 勿下调
RANSAC_K = 15                     # 候选平面个数: 多种子轮流取最大平面, 防被墙/地吸走
RANSAC_MIN_IN = 300               # 单候选最少内点数(降采样系), 低于则停止取候选
RANSAC_ITERS = 2000               # 单候选 RANSAC 迭代数
# 平面精修(_refine_plane, 滑窗之前)
REFINE_ITERS = 3                  # SVD 重拟合轮数
REFINE_K = 2.0                    # 剔除阈值 = max(12mm, REFINE_K x 中位距离)
REFINE_FLOOR = 0.012              # 剔除阈值下限 12mm
REFINE_MIN_KEEP = 300             # 精修保留点数下限, 低于则停止迭代
VERTICAL_NZ_MAX = 0.5             # 候选 |n_z| 上限: 跳过地板/天花板(板近似竖直)
# 两段式滑窗
WIN_CELL = 0.01                   # 滑窗占据栅格边长 1cm
RING_CELLS = 3                    # 环带宽度 = 3 格 = 3cm(窗外紧贴一圈, 真板此处应空)
RING_PENALTY = 3.0                # 评分 = 窗内点 - RING_PENALTY x 环带点
ANGLE_COARSE_STEP = 6.0           # 粗扫角度步长(全 ±90°)
ANGLE_FINE_RANGE = 6.0            # 细扫角度范围(粗赢家 ±6°)
ANGLE_FINE_STEP = 2.0             # 细扫角度步长
MIN_WIN_CNT = 300                 # 窗内点下限(降采样系), 低于则该候选弃用
MIN_SEL = 300                     # 最终选中点下限(原始点云系), 低于则整帧 FAIL
# 带内最大连通域(滑窗后的"原平面"口径)
BAND_CELL = 0.02                  # 连通域栅格边长 2cm
# 质量门(与 calibrate_camera_lidar.py 顶部的门同步修改):
#   裁剪长边 >= 0.78m, 短边 >= 0.54m, 尺寸误差 <= 16%, 原占比 >= 0.85, 否则整帧剔除


def _ransac_planes(P: np.ndarray, dist_thr: float = DIST_THR, K: int = RANSAC_K,
                   min_in: int = RANSAC_MIN_IN, iters: int = RANSAC_ITERS,
                   seed: int = 0):
    """文档 §3.2 方案A: 确定性多种子 RANSAC, 每轮取最大平面并剔除内点。"""
    rng = np.random.default_rng(seed)
    rest = np.ones(len(P), bool)
    cands = []
    for _ in range(K):
        idx = np.where(rest)[0]
        if len(idx) < min_in:
            break
        Q = P[idx]
        best = None
        for _ in range(iters):
            i, j, k = rng.choice(len(Q), 3, replace=False)
            nv = np.cross(Q[j] - Q[i], Q[k] - Q[i])
            L = np.linalg.norm(nv)
            if L < 1e-9:
                continue
            nv /= L
            dd = float(nv @ Q[i])
            c = int((np.abs(Q @ nv - dd) < dist_thr).sum())
            if best is None or c > best[0]:
                best = (c, nv, dd)
        if best is None or best[0] < min_in:
            break
        _, nv, dd = best
        inl = np.abs(Q @ nv - dd) < dist_thr
        cands.append((nv, dd, idx[inl]))
        rest[idx[inl]] = False
    return cands



def _refine_plane(pts: np.ndarray, n0: np.ndarray, d0: float,
                  iters: int = REFINE_ITERS, k: float = REFINE_K):
    """迭代精修平面: SVD 重拟合 + 按 max(REFINE_FLOOR, k*中位距离) 剔离面散点(如板顶小条)。
    返回 (n, d, kept_points)。"""
    n = n0.copy()
    d = d0
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        dist = np.abs(pts @ n - d)
        med = float(np.median(dist)) if len(dist) else 0.0
        thr = max(REFINE_FLOOR, k * med)
        keep = dist < thr
        if keep.sum() < REFINE_MIN_KEEP:
            break
        sel = pts[keep]
        c = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        nn = vt[2]
        if nn @ n < 0:
            nn = -nn
        n = nn
        d = float(nn @ c)
    return n, d, pts[keep]


def _remove_ground(P: np.ndarray, thr: float = GROUND_THR, iters: int = GROUND_ITERS,
                   nz_min: float = GROUND_NZ_MIN, seed: int = 0):
    """RANSAC 最强水平面(地板)内点剔除。天花板等远处水平面由后续最大连通域处理。"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j, k = rng.choice(len(P), 3, replace=False)
        nv = np.cross(P[j] - P[i], P[k] - P[i])
        L = np.linalg.norm(nv)
        if L < 1e-9:
            continue
        nv /= L
        if abs(nv[2]) < nz_min:
            continue
        dd = float(nv @ P[i])
        cnt = int((np.abs(P @ nv - dd) < thr).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nv, dd)
    if best is None:
        return np.ones(len(P), bool)
    return np.abs(P @ best[1] - best[2]) >= thr


def _band_cc(Pg: np.ndarray, info: dict, band: np.ndarray, cell: float = BAND_CELL) -> np.ndarray:
    """slab(|n·x-d|<thr)无限延伸会吞进地面/天花板/散点; 带内点投影到平面2D栅格后
    仅保留最大连通域, 与板不连通的杂点全部排除。"""
    pts = Pg[band]
    if len(pts) < 10:
        return band
    u, v = (pts - info["c"]) @ info["e1"], (pts - info["c"]) @ info["e2"]
    iu = ((u - u.min()) / cell).astype(int)
    iv = ((v - v.min()) / cell).astype(int)
    img = np.zeros((iv.max() + 1, iu.max() + 1), np.uint8)
    img[iv, iu] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    if count <= 2:
        return band
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = labels[iv, iu] == largest
    out = np.zeros(len(Pg), bool)
    out[np.where(band)[0][sel]] = True
    return out


def extract_rect_plane(P: np.ndarray, L: float = BOARD_L, W: float = BOARD_W,
                       voxel: float = VOXEL, K: int = RANSAC_K, seed: int = 0):
    """文档 §3 管线(确定性): 去地面 -> 多种子RANSAC候选(±DIST_THR) -> 平面SVD精修
    -> 2D滑已知尺寸窗(环带评分) -> 带内最大连通域 -> 回原始点云。"""
    dist_thr = DIST_THR
    keep = _remove_ground(P)
    Pg = P[keep]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pg)
    ds = np.asarray(pcd.voxel_down_sample(voxel).points)

    def best_window(u, v, cell=WIN_CELL, ring_cells=RING_CELLS):
        """滑已知 LxW 窗, 角度 -90~90(粗扫+细扫)。
        评分 = 窗内点 - RING_PENALTY x 边缘环带点: 真板窗外的环带几乎为空。"""
        kw, kh = int(round(L / cell)), int(round(W / cell))
        R = ring_cells

        def search(angles):
            best = None                       # (score, win, wc, hc, ang)
            for a in angles:
                r = np.radians(a)
                ca, sa = np.cos(r), np.sin(r)
                uu = u * ca + v * sa
                vv = -u * sa + v * ca
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
                               - I[pr + R + kh, pc + R] + I[pr + R, pc + R])
                        big = (I[pr + 2 * R + kh, pc + 2 * R + kw] - I[pr, pc + 2 * R + kw]
                               - I[pr + 2 * R + kh, pc] + I[pr, pc])
                        ring = big - win
                        score = win - RING_PENALTY * ring
                        if best is None or score > best[0]:
                            best = (score, win, ulo + (pc + kw / 2) * cell,
                                    vlo + (pr + kh / 2) * cell, a)
            return best

        coarse = search(np.arange(-90, 90.1, ANGLE_COARSE_STEP))
        return search(np.arange(coarse[4] - ANGLE_FINE_RANGE,
                                coarse[4] + ANGLE_FINE_RANGE + 0.1, ANGLE_FINE_STEP))

    cands = []
    for nv, dd, idx in _ransac_planes(ds, dist_thr, K=K, seed=seed):
        n = nv / np.linalg.norm(nv)
        if abs(n[2]) > VERTICAL_NZ_MAX:
            continue                        # 地板/天花板, 不可能是竖直的板
        n_r, d_r, pts = _refine_plane(ds[idx], n, dd)
        if d_r < 0:                     # 统一法向约定: 与相机PnP一致(背离传感器)
            n_r, d_r = -n_r, -d_r
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        e1 = vt[0] - np.dot(vt[0], n_r) * n_r
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(n_r, e1)
        u, v = (pts - c) @ e1, (pts - c) @ e2
        res = best_window(u, v)
        if res is None:
            continue
        score, cnt, wc, hc, ang = res
        if cnt < MIN_WIN_CNT:
            continue
        outside = 1.0 - cnt / len(idx)
        cands.append(dict(score=score, cnt=int(cnt), total=len(idx), outside=outside,
                          n=n_r, d=d_r, c=c, e1=e1, e2=e2, wc=wc, hc=hc, ang=ang))

    valid = [c for c in cands if c["cnt"] >= 2 * MIN_WIN_CNT]
    if not valid:
        return None, None
    best = max(valid, key=lambda c: c["score"])

    # 3.5 回原始点云(去地面后): 平面距离 + 旋转已知尺寸窗 SDF + 带内最大连通域
    th = np.radians(-best["ang"])
    ca, sa = np.cos(th), np.sin(th)
    u = (Pg - best["c"]) @ best["e1"]
    v = (Pg - best["c"]) @ best["e2"]
    du, dv = u - best["wc"], v - best["hc"]
    uu = du * ca - dv * sa
    vv = du * sa + dv * ca
    band_g = np.abs(Pg @ best["n"] - best["d"]) < dist_thr
    cc_g = _band_cc(Pg, best, band_g)
    mask_g = band_g & cc_g \
        & (np.abs(uu) <= L / 2 + voxel) & (np.abs(vv) <= W / 2 + voxel)
    if mask_g.sum() < MIN_SEL:
        return None, None
    mask = np.zeros(len(P), bool)
    mask[np.where(keep)[0]] = mask_g
    band_full = np.zeros(len(P), bool)
    band_full[np.where(keep)[0]] = cc_g
    best["n_sel"] = int(mask_g.sum())
    best["band_cc_n"] = int(cc_g.sum())
    best["band_mask"] = band_full
    return mask, best


def write_ply(path, pts, colors=None):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            np.savetxt(f, pts, fmt="%.6f")
        else:
            np.savetxt(f, np.column_stack([pts, colors.astype(float)]),
                       fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])


def main() -> int:
    ap = argparse.ArgumentParser(description="按指南管线分割 0.84x0.60 矩形板")
    ap.add_argument("paths", nargs="+", help="帧目录或 merged.ply")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print(f"{'帧':<12}{'窗内点':>8}{'实测尺寸cm':>14}{'尺寸误差':>9}{'环带分':>9}{'选中':>8}{'原占比':>7}  警告")
    for p in a.paths:
        d = p if os.path.isdir(p) else os.path.dirname(p)
        mp = os.path.join(d, "merged.ply")
        if not os.path.exists(mp):
            print(f"  {os.path.basename(d):<10} 缺 merged.ply, 跳过")
            continue
        with open(mp) as f:
            for i, line in enumerate(f, 1):
                if line.strip() == "end_header":
                    break
        P = np.loadtxt(mp, skiprows=i)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        P = P[:, :3]
        mask, info = extract_rect_plane(P)
        name = os.path.basename(d)
        if mask is None:
            print(f"  {name:<10}  FAIL (无合格候选平面)")
            continue
        sel = P[mask]
        c0 = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - c0, full_matrices=False)
        uv = np.stack([(sel - c0) @ vt[0], (sel - c0) @ vt[1]], axis=1).astype(np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(uv)
        el, ew = max(rw, rh) * 100, min(rw, rh) * 100
        se = abs(el - 84) / 84 + abs(ew - 60) / 60
        band_n = int(info.get("band_cc_n", 0))
        ratio_P = info["n_sel"] / max(band_n, 1)
        warn = ""
        if ratio_P < 0.85:
            warn += " [占比低?]"
        if se > 0.16:
            warn += " [缩窗?]"
        print(f"  {name:<10}{info['cnt']:>8}{el:>6.1f}x{ew:>5.1f}{se * 100:>8.1f}%"
              f"{info['score']:>9.0f}{info['n_sel']:>8}{ratio_P:>7.2f}{warn}")
        if a.save:
            write_ply(os.path.join(d, "board_rect.ply"), sel)
            band_full = info.get("band_mask")
            if band_full is None:
                band_full = np.abs(P @ info["n"] - info["d"]) < DIST_THR
            cols = np.full((len(P), 3), 120, np.uint8)
            cols[band_full & ~mask] = (30, 30, 255)   # 带内未选中(最大联通域内) = 蓝
            cols[mask] = (255, 30, 30)                # 选中 = 红
            write_ply(os.path.join(d, "board_rect_overlay.ply"), P, cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
