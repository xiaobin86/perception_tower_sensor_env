#!/usr/bin/env python3
"""实验: 窗口评分改成 纯 max(win)(不要 ring 惩罚), 看窗口是否回到真板位置。

动机: 板是平面上最致密的区域, "窗内点数最多"的位置应该就是板。
方法: 复用与生产完全相同的候选平面/栅格/积分图, 只把 score = win - 3*ring
改成 score = win(隔离变量)。与相机 PnP 真板位置和生产版窗口三方对比。
不改动 segment_board.py 的任何逻辑。
"""

import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open3d as o3d
import segment_board as sb
from calibrate_camera_lidar import (BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_X,
                                    SQUARE_SIZE_Y, board_object_points,
                                    find_board_corners, load_camera_info,
                                    solve_pnp_robust)

POSES = ["turntable_output/20260923_072321", "turntable_output/20260923_072547"]
L, W, CELL, R = sb.BOARD_L, sb.BOARD_W, sb.WIN_CELL, sb.RING_CELLS
KW, KH = int(round(L / CELL)), int(round(W / CELL))


def read_ply_xyz(path):
    with open(path) as f:
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        return np.loadtxt(f, max_rows=n)[:, :3].astype(np.float64)


def camera_truth(pose_dir):
    """相机 PnP 板中心 -> merge 系(Rz(-90) photo->merge)。"""
    K, dist = load_camera_info("config/camera_info.yaml")
    img = cv2.imread(os.path.join(pose_dir, "color.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(
        gray, find_board_corners(img), (5, 5), (-1, -1), crit).reshape(-1, 2)
    rvec, tvec, *_ = solve_pnp_robust(board_object_points(), corners, K, dist)
    Rc, _ = cv2.Rodrigues(rvec)
    obj_c = np.array([(BOARD_COLS - 1) / 2 * SQUARE_SIZE_X,
                      (BOARD_ROWS - 1) / 2 * SQUARE_SIZE_Y, 0.0])
    center_cam = Rc @ obj_c + tvec.ravel()
    ext = yaml.safe_load(open("config/camera_extrinsics.yaml"))["lidar_to_camera"]
    R = np.array(ext["rotation_matrix"]).reshape(3, 3)
    t = np.array(ext["translation"])
    c_photo = R.T @ (center_cam - t)
    a = np.radians(-90.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ c_photo


def window_search(u, v, score_mode):
    """与生产相同的粗扫+细扫, 评分可选 'win' 或 'win-3ring'。返回 (score,win,ring,wc,hc,ang)。"""
    def search(angles):
        best = None
        for ang in angles:
            r = np.radians(ang)
            ca, sa = np.cos(r), np.sin(r)
            uu = u * ca + v * sa
            vv = -u * sa + v * ca
            ulo, vlo = uu.min() - 0.05, vv.min() - 0.05
            nu = int((uu.max() - ulo) / CELL) + 3
            nv = int((vv.max() - vlo) / CELL) + 3
            G = np.zeros((nv + 2 * R, nu + 2 * R))
            iu = np.clip(((uu - ulo) / CELL).astype(int) + R, 0, nu + 2 * R - 1)
            iv = np.clip(((vv - vlo) / CELL).astype(int) + R, 0, nv + 2 * R - 1)
            np.add.at(G, (iv, iu), 1.0)
            I = np.pad(G.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            for pr in range(0, nv - KH + 1):
                for pc in range(0, nu - KW + 1):
                    win = (I[pr+R+KH, pc+R+KW] - I[pr+R, pc+R+KW]
                           - I[pr+R+KH, pc+R] + I[pr+R, pc+R])
                    big = (I[pr+2*R+KH, pc+2*R+KW] - I[pr, pc+2*R+KW]
                           - I[pr+2*R+KH, pc] + I[pr, pc])
                    ring = big - win
                    s = win if score_mode == "win" else win - sb.RING_PENALTY * ring
                    if best is None or s > best[0]:
                        best = (s, win, ring, ulo + (pc + KW / 2) * CELL,
                                vlo + (pr + KH / 2) * CELL, ang)
        return best
    coarse = search(np.arange(-90, 90.1, sb.ANGLE_COARSE_STEP))
    return search(np.arange(coarse[5] - sb.ANGLE_FINE_RANGE,
                            coarse[5] + sb.ANGLE_FINE_RANGE + 0.1, sb.ANGLE_FINE_STEP))


def get_winning_candidate(P):
    """与生产相同地选出胜出候选平面, 返回 (u, v, c, n)。"""
    keep = sb._remove_ground(P)
    Pg = P[keep]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pg)
    ds = np.asarray(pcd.voxel_down_sample(sb.VOXEL).points)
    cands = []
    for nv_, dd, idx in sb._ransac_planes(ds, sb.DIST_THR, K=sb.RANSAC_K, seed=0):
        n = nv_ / np.linalg.norm(nv_)
        if abs(n[2]) > sb.VERTICAL_NZ_MAX:
            continue
        n_r, d_r, pts = sb._refine_plane(ds[idx], n, dd)
        if d_r < 0:
            n_r, d_r = -n_r, -d_r
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        e1 = vt[0] - np.dot(vt[0], n_r) * n_r
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(n_r, e1)
        u, v = (pts - c) @ e1, (pts - c) @ e2
        res = window_search(u, v, "win-3ring")
        if res and res[1] >= sb.MIN_WIN_CNT:
            cands.append(dict(score=res[0], u=u, v=v, c=c, e1=e1, e2=e2))
    cands.sort(key=lambda x: -x["score"])
    return cands[0]


def win_center_3d(cand, wc, hc, ang):
    """窗口中心(wc,hc,ang 是旋转坐标系中的中心)-> 3D(merge 系)。"""
    th = np.radians(ang)
    ca, sa = np.cos(th), np.sin(th)
    A = np.array([[ca, sa], [-sa, ca]])      # uu = u*ca+v*sa, vv = -u*sa+v*ca
    u0, v0 = np.linalg.solve(A, np.array([wc, hc]))
    return cand["c"] + u0 * cand["e1"] + v0 * cand["e2"]


def filter_largest_cc(u, v):
    """平面内点栅格化 -> 最大连通域 -> 仅保留域内点(剔除板外共面条带/散点)。"""
    import cv2
    ulo, vlo = u.min() - 0.05, v.min() - 0.05
    nu = int((u.max() - ulo) / CELL) + 3
    nv = int((v.max() - vlo) / CELL) + 3
    G = np.zeros((nv, nu), np.uint8)
    iu = ((u - ulo) / CELL).astype(int).clip(0, nu - 1)
    iv = ((v - vlo) / CELL).astype(int).clip(0, nv - 1)
    G[iv, iu] = 1
    ncc, labels = cv2.connectedComponents(G, connectivity=8)
    if ncc <= 1:
        return u, v, 1.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    keep = labels[iv, iu] == largest
    return u[keep], v[keep], keep.mean()


def compare_pose(pose_dir):
    print(f"\n=== {os.path.basename(pose_dir)} ===")
    P = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))
    truth = camera_truth(pose_dir)
    cand = get_winning_candidate(P)

    u_f, v_f, ratio = filter_largest_cc(cand["u"], cand["v"])
    print(f"  平面内点 {len(cand['u'])} -> 最大连通域 {len(u_f)} (保留 {ratio:.2%})")

    res_prod = window_search(cand["u"], cand["v"], "win-3ring")
    res_pure = window_search(cand["u"], cand["v"], "win")
    res_cc3 = window_search(u_f, v_f, "win-3ring")
    res_ccw = window_search(u_f, v_f, "win")
    c_prod = win_center_3d(cand, res_prod[3], res_prod[4], res_prod[5])
    c_pure = win_center_3d(cand, res_pure[3], res_pure[4], res_pure[5])
    c_cc3 = win_center_3d(cand, res_cc3[3], res_cc3[4], res_cc3[5])
    c_ccw = win_center_3d(cand, res_ccw[3], res_ccw[4], res_ccw[5])

    def show(tag, res, c3d):
        d = np.linalg.norm(c3d - truth)
        print(f"  {tag:<18} win={res[1]:6.0f} ring={res[2]:5.0f} 中心=({res[3]:+.3f},{res[4]:+.3f})"
              f" ang={res[5]:+.0f}  距真板={d*100:5.1f}cm")
        return d

    print(f"  真板中心(相机PnP): {truth.round(3)}")
    d_prod = show("win-3ring(原始)", res_prod, c_prod)
    d_pure = show("max-win(原始)", res_pure, c_pure)
    d_cc3 = show("win-3ring(最大CC)", res_cc3, c_cc3)
    d_ccw = show("max-win(最大CC)", res_ccw, c_ccw)
    return d_prod, d_pure, d_cc3, d_ccw


if __name__ == "__main__":
    results = {}
    for pose in POSES:
        results[pose] = compare_pose(pose)
    print("\n=== 汇总: 距真板偏差(cm), 越小越好 ===")
    print(f"{'帧':<22}{'3ring原始':>10}{'maxwin原始':>11}{'3ring最大CC':>12}{'maxwin最大CC':>13}")
    for pose, ds in results.items():
        print(f"{os.path.basename(pose):<22}" + "".join(f"{d*100:>10.1f}" if i == 0 else
              f"{d*100:>11.1f}" if i == 1 else f"{d*100:>12.1f}" if i == 2 else f"{d*100:>13.1f}"
              for i, d in enumerate(ds)))

