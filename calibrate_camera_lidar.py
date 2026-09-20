#!/usr/bin/env python3
"""Camera-LiDAR extrinsic calibration from a planar checkerboard.

Each pose directory must contain `color.png` and `merged.ply`. See AGENTS.md for the
coordinate frame conventions.

Pipeline per pose:
    image  -> findChessboardCorners(4x6) -> solvePnP -> camera plane (n_c, d_c)
    cloud  -> remove ground -> search board-like plane -> LiDAR plane (n_l, d_l)
Then:  n_c = R n_l   and   t from board centers, solved over all poses.

Usage:
    python3 calibrate_camera_lidar.py POSE_DIR [POSE_DIR ...] [--camera-info config/camera_info.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass, replace

import cv2
import numpy as np
import yaml

SQUARE_SIZE_M = 0.116  # 实测格距(标称120mm, 尺子量得116, 见 docs/tz_bias_analysis.md 尺度偏差溯源)
BOARD_COLS = 4
BOARD_ROWS = 6
BOARD_WIDTH_M = 0.60
BOARD_HEIGHT_M = 0.84
BOARD_TRIM_TOL_M = 0.02
GROUND_MAX_TILT_COS = 0.96
GROUND_MIN_SPAN_M = 1.2
GROUND_SLAB_M = 0.01
MIN_BOARD_INLIERS = 2000
SELF_RETURN_RADIUS_M = 0.25
PLANE_INLIER_M = 0.02


@dataclass(frozen=True, slots=True)
class Plane:
    normal: np.ndarray
    offset: float


@dataclass(frozen=True, slots=True)
class PoseObs:
    name: str
    pose_dir: str
    camera_plane: Plane
    lidar_plane: Plane
    image: np.ndarray
    image_points: np.ndarray
    lidar_inliers: np.ndarray
    up_cam: np.ndarray
    center_cam: np.ndarray
    center_lidar: np.ndarray
    board_polygon: np.ndarray


def board_object_points() -> np.ndarray:
    pts = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            pts.append([col * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0.0])
    return np.array(pts, dtype=np.float64)


def load_camera_info(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        info = yaml.safe_load(f)
    K = np.array(info["k"], dtype=np.float64).reshape(3, 3)
    dist = np.array(info["d"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def read_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for header_lines, line in enumerate(f, start=1):
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path}: no 'end_header' (not ASCII PLY?)")
    points = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    return points[:, :3]


def write_ply_xyz(path: str, points: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.6f")


def fit_plane_svd(points: np.ndarray) -> Plane:
    centroid = points.mean(axis=0)
    cov = (points - centroid).T @ (points - centroid)
    _, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    offset = float(normal @ centroid)
    if offset < 0.0:
        normal, offset = -normal, -offset
    return Plane(normal, offset)


def ransac_plane(points: np.ndarray, rng: np.random.Generator,
                 threshold: float = PLANE_INLIER_M, iters: int = 3000) -> Plane:
    best_count, best = -1, None
    n = len(points)
    for _ in range(iters):
        a, b, c = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = float(normal @ a)
        count = int((np.abs(points @ normal - offset) < threshold).sum())
        if count > best_count:
            best_count, best = count, Plane(normal, offset)
    assert best is not None
    inliers = points[np.abs(points @ best.normal - best.offset) < threshold]
    return fit_plane_svd(inliers)


def find_ground_plane(points: np.ndarray, rng: np.random.Generator,
                      rounds: int = 6, iters: int = 1500, threshold: float = 0.03,
                      sample_max: int = 60000) -> Plane:
    """RANSAC 迭代找大面积近水平平面（地板），不依赖 z 轴朝向；不合格的大平面先剔除。"""
    rest = points
    for _ in range(rounds):
        if len(rest) < 1000:
            break
        sample = rest
        if len(rest) > sample_max:
            sample = rest[rng.choice(len(rest), sample_max, replace=False)]
        best = None
        for _ in range(iters):
            a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            if abs(normal[2]) < GROUND_MAX_TILT_COS:
                continue
            offset = float(normal @ a)
            count = int((np.abs(sample @ normal - offset) < threshold).sum())
            if best is None or count > best[0]:
                best = (count, normal, offset)
        if best is None:
            break
        _, normal, offset = best
        inliers = rest[np.abs(rest @ normal - offset) < threshold]
        spans = _plane_spans(inliers)
        if spans[1] >= GROUND_MIN_SPAN_M and spans[2] >= GROUND_MIN_SPAN_M:
            return fit_plane_svd(inliers)
        rest = rest[np.abs(rest @ normal - offset) >= threshold]
    raise ValueError("no large horizontal plane found")


def _plane_spans(inliers: np.ndarray) -> np.ndarray:
    c = inliers.mean(axis=0)
    cov = (inliers - c).T @ (inliers - c)
    _, vecs = np.linalg.eigh(cov)
    spans = []
    for k in range(3):
        proj = (inliers - c) @ vecs[:, k]
        spans.append(proj.max() - proj.min())
    return np.sort(spans)


def _is_board_plane(inliers: np.ndarray, normal: np.ndarray) -> bool:
    if len(inliers) < 300 or abs(normal[2]) > 0.85:
        return False
    s = _plane_spans(inliers)
    return bool(s[0] < 0.30 and 0.30 < s[1] < 1.20 and 0.45 < s[2] < 3.00)


def quad_overlap_score(inliers: np.ndarray, quad: np.ndarray, K: np.ndarray, dist: np.ndarray,
                       R: np.ndarray, t: np.ndarray, photo_angle: float) -> float:
    if len(inliers) < 200:
        return 0.0
    Rz = rotation_z(photo_angle)
    rvec = cv2.Rodrigues(R @ Rz)[0]
    proj, _ = cv2.projectPoints(inliers, rvec, t, K, dist)
    proj = proj.reshape(-1, 2)
    pc = (R @ (inliers @ Rz.T).T).T + t
    ok = np.isfinite(proj).all(axis=1) & (pc[:, 2] > 0.2)
    if int(ok.sum()) < 100:
        return 0.0
    poly = np.asarray(quad, dtype=np.float32)
    inside = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), True) >= 0.0
                       for u, v in proj[ok]])
    return float(inside.mean())


def search_board_plane(points: np.ndarray, rng: np.random.Generator, rounds: int = 8,
                       quad: np.ndarray | None = None, K: np.ndarray | None = None,
                       dist: np.ndarray | None = None, init: tuple[np.ndarray, np.ndarray] | None = None,
                       photo_angle: float = 90.0) -> tuple[Plane, np.ndarray]:
    candidates = []
    rest = points
    for _ in range(rounds):
        if len(rest) < 500:
            break
        plane = ransac_plane(rest, rng)
        inliers = rest[np.abs(rest @ plane.normal - plane.offset) < PLANE_INLIER_M]
        candidates.append((plane, inliers))
        keep = np.abs(rest @ plane.normal - plane.offset) >= PLANE_INLIER_M
        if keep.sum() < 500:
            break
        rest = rest[keep]
    gated = [(plane, inliers) for plane, inliers in candidates
             if _is_board_plane(inliers, plane.normal)]
    if quad is not None and init is not None:
        pool = gated if gated else candidates
        best = None
        for plane, inliers in pool:
            score = quad_overlap_score(inliers, quad, K, dist, init[0], init[1], photo_angle)
            if best is None or score > best[0]:
                best = (score, plane, inliers)
        if best is not None and best[0] > 0.3:
            return best[1], best[2]
    if gated:
        return gated[0]
    raise ValueError("no board-like plane found")


def clip_to_board_quad(inliers: np.ndarray, quad: np.ndarray, K: np.ndarray, dist: np.ndarray,
                       init: tuple[np.ndarray, np.ndarray], photo_angle: float,
                       margin_px: float = -6.0) -> np.ndarray:
    Rz = rotation_z(photo_angle)
    rvec = cv2.Rodrigues(init[0] @ Rz)[0]
    proj, _ = cv2.projectPoints(inliers, rvec, init[1], K, dist)
    proj = proj.reshape(-1, 2)
    pc = (init[0] @ (inliers @ Rz.T).T).T + init[1]
    ok = np.isfinite(proj).all(axis=1) & (pc[:, 2] > 0.2)
    poly = np.asarray(quad, dtype=np.float32)
    keep = np.array([ok[i] and cv2.pointPolygonTest(poly, (float(u), float(v)), True) >= margin_px
                     for i, (u, v) in enumerate(proj)])
    return inliers[keep]


def guided_board_plane(points: np.ndarray, camera_plane: Plane, init: tuple[np.ndarray, np.ndarray],
                       photo_angle: float, band_m: float = 0.20, fit_m: float = 0.015,
                       iters: int = 3) -> tuple[Plane, np.ndarray] | None:
    """用相机板平面 + 初始外参预测雷达系板平面，在其附近带内拟合（不依赖 RANSAC 抽签）。"""
    R0, t0 = init
    n_c, d_c = camera_plane.normal, camera_plane.offset
    n_photo = R0.T @ n_c
    d_photo = d_c - float(n_c @ t0)
    Rz = rotation_z(photo_angle)
    normal = Rz.T @ n_photo
    offset = d_photo
    band = points[np.abs(points @ normal - offset) < band_m]
    if len(band) < 300:
        return None
    plane = None
    for _ in range(iters):
        plane = fit_plane_svd(band)
        if float(plane.normal @ normal) < 0.0:
            plane = Plane(-plane.normal, -plane.offset)
        band = points[np.abs(points @ plane.normal - plane.offset) < fit_m]
        if len(band) < 300:
            return None
    if plane is None:
        return None
    return fit_plane_svd(band), band


def extract_lidar_board(points: np.ndarray, rng: np.random.Generator, out_dir: str,
                        camera_plane: Plane | None = None, quad: np.ndarray | None = None,
                        K: np.ndarray | None = None,
                        dist: np.ndarray | None = None,
                        init: tuple[np.ndarray, np.ndarray] | None = None,
                        photo_angle: float = 90.0,
                        use_guided: bool = False) -> tuple[Plane, np.ndarray]:
    try:
        from segment_board import extract_rect_plane
        res = extract_rect_plane(points)
    except Exception as exc:
        print(f"    [board] segment_board 不可用({exc})，回退旧路径")
        res = (None, None)
    if res[0] is not None:
        mask, info = res
        pts = points[mask]
        if len(pts) >= MIN_BOARD_INLIERS:
            c0 = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
            uv = np.stack([(pts - c0) @ vt[0], (pts - c0) @ vt[1]], axis=1).astype(np.float32)
            _, (rw, rh), _ = cv2.minAreaRect(uv)
            lo, sh = max(rw, rh), min(rw, rh)
            band_n = max(int(info.get("band_cc_n", 1)), 1)
            ratio = len(pts) / band_n
            se = abs(lo - 0.84) / 0.84 + abs(sh - 0.60) / 0.60
            # 质量门: 裁剪应接近满尺寸 0.84x0.60(下限+误差上限) 且最大联通区域
            # 占比足够(混墙/分割错 -> 占比低; 条带/缩窗 -> 尺寸不对)
            if lo < 0.78 or sh < 0.54 or se > 0.16 or ratio < 0.85:
                raise ValueError(
                    f"board quality gate: size {lo*100:.0f}x{sh*100:.0f}cm "
                    f"err {se*100:.0f}% ratio {ratio:.2f}")
            print(f"    [board] segment_board ✓ 实测 {lo*100:.1f}×{sh*100:.1f} cm"
                  f"(期望 84×60) 占比 {ratio:.2f} → {len(pts)} 点")
            write_ply_xyz(os.path.join(out_dir, "dbg_3_plane.ply"), pts)
            return Plane(normal=info["n"], offset=info["d"]), pts
        print(f"    [board] segment_board 只取到 {len(pts)} 点，回退旧路径")
    else:
        print("    [board] segment_board 未找到，回退旧路径")

    ground = find_ground_plane(points, rng)
    radius = np.linalg.norm(points, axis=1)
    scene = points[(np.abs(points @ ground.normal - ground.offset) > GROUND_SLAB_M)
                   & (radius > SELF_RETURN_RADIUS_M)]
    write_ply_xyz(os.path.join(out_dir, "dbg_2_no_ground.ply"), scene)
    guided = None
    if use_guided and camera_plane is not None and init is not None:
        guided = guided_board_plane(scene, camera_plane, init, photo_angle)
    if guided is not None:
        plane, inliers = guided
        if quad is not None:
            inliers = clip_to_board_quad(inliers, quad, K, dist, init, photo_angle)
        window = board_window_mask(inliers, plane.normal)
        if window is not None and int(window.sum()) >= 200:
            inliers = inliers[window]
    else:
        plane, inliers = search_board_plane(scene, rng)
    rough = fit_plane_svd(inliers)
    comp = largest_component_mask(inliers, rough.normal)
    if comp is not None and int(comp.sum()) >= 200:
        inliers = inliers[comp]
        plane = fit_plane_svd(inliers)
    write_ply_xyz(os.path.join(out_dir, "dbg_3_plane.ply"), inliers)
    if len(inliers) < MIN_BOARD_INLIERS:
        raise ValueError(f"board patch too sparse ({len(inliers)} points < {MIN_BOARD_INLIERS})")
    return plane, inliers


def _try_detect(gray, pat, flags):
    try:
        ok, corners = cv2.findChessboardCornersSB(gray, pat, flags=flags)
        if ok:
            return corners
    except cv2.error:
        pass
    return None


def find_board_corners(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns = [(BOARD_COLS, BOARD_ROWS), (BOARD_ROWS, BOARD_COLS)]
    # 先用 OpenCV 现代检测器 SB（自带亚像素，定位明显更准），失败再退回经典检测器
    for pat in patterns:
        try:
            ok, corners = cv2.findChessboardCornersSB(
                gray, pat, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
            if ok:
                return corners
        except cv2.error:
            pass
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for pat in patterns:
        found, corners = cv2.findChessboardCorners(gray, pat, flags)
        if found:
            return corners
    # 暖色低对比度印刷会击穿自适应阈值，需要多组窗口大小；Otsu 与 B/G 通道作为补充
    thresh_variants = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block, c)
        for block, c in ((21, 10), (21, 5), (31, 7), (31, 10), (41, 7), (41, 5))
    ] + [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]
    channels = [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 1],
                cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 2]]
    for th in thresh_variants:
        for pat in patterns:
            try:
                ok, corners = cv2.findChessboardCornersSB(th, pat)
                if ok:
                    return corners
            except cv2.error:
                pass
    for ch in channels:
        for pat in patterns:
            try:
                ok, corners = cv2.findChessboardCornersSB(ch, pat)
                if ok:
                    return corners
            except cv2.error:
                pass
    raise ValueError(f"checkerboard ({BOARD_COLS}x{BOARD_ROWS} inner corners) not found")


def solve_pnp_robust(object_points: np.ndarray, corners: np.ndarray, K: np.ndarray,
                     dist: np.ndarray, max_bad: int = 2, thr: float = 2.5,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, bool]:
    """solvePnP(IPPE+LM) 带坏角点剔除: 重投影误差>thr 的最坏角点逐轮剔除(最多 max_bad 个)。

    为什么需要: 单个坏角点(检测毛刺)可把整帧位姿带偏(实测 0.5°/9mm)。
    坏角点超过 max_bad 个时不抛错, 由调用方决定(truncated=True 表示仍有坏角点未剔)。
    返回 (rvec, tvec, kept_idx(原始编号), rejected=[(原始编号, 误差px)], truncated)。
    """
    obj_pts = np.asarray(object_points, dtype=np.float64)
    img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    kept = np.arange(len(img_pts))
    rejected: list = []
    rvec = tvec = None
    truncated = False
    for _round in range(max_bad + 1):
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
        if not ok:
            raise ValueError("solvePnP failed")
        reproj = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)[0].reshape(-1, 2)
        err = np.linalg.norm(reproj - img_pts, axis=1)
        if err.max() < thr or len(img_pts) <= 8:
            break
        if _round == max_bad:
            truncated = True
            break
        worst = int(np.argmax(err))
        rejected.append((int(kept[worst]), float(err[worst])))
        mask = np.ones(len(img_pts), bool)
        mask[worst] = False
        obj_pts, img_pts, kept = obj_pts[mask], img_pts[mask], kept[mask]
    return rvec, tvec, kept, rejected, truncated


def detect_camera_plane(image: np.ndarray, K: np.ndarray, dist: np.ndarray
                        ) -> tuple[Plane, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_all = cv2.cornerSubPix(
        gray, find_board_corners(image), (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    ).reshape(-1, 2)
    rvec, tvec, kept, rejected, truncated = solve_pnp_robust(
        board_object_points(), corners_all, K, dist)
    for idx, e in rejected:
        print(f"    [camera] 剔除坏角点 帧内#{idx} (重投影误差 {e:.1f}px)")
    if truncated:
        raise ValueError("坏角点超过2个, 整帧剔除")
    corners = corners_all[kept]
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]
    t = tvec.ravel()
    offset = float(normal @ t)
    if offset < 0.0:
        normal, offset = -normal, -offset
    board_center = R @ np.array([(BOARD_COLS - 1) / 2 * SQUARE_SIZE_M,
                                 (BOARD_ROWS - 1) / 2 * SQUARE_SIZE_M, 0.0]) + t
    paper = np.array([
        [-SQUARE_SIZE_M, -SQUARE_SIZE_M, 0.0],
        [BOARD_COLS * SQUARE_SIZE_M, -SQUARE_SIZE_M, 0.0],
        [BOARD_COLS * SQUARE_SIZE_M, BOARD_ROWS * SQUARE_SIZE_M, 0.0],
        [-SQUARE_SIZE_M, BOARD_ROWS * SQUARE_SIZE_M, 0.0],
    ])
    polygon, _ = cv2.projectPoints(paper, rvec, tvec, K, dist)
    return (Plane(normal, offset), corners.reshape(-1, 2), -R[:, 1], board_center,
            polygon.reshape(-1, 2))


def solve_rotation(lidar_vecs: np.ndarray, camera_vecs: np.ndarray) -> np.ndarray:
    H = lidar_vecs @ camera_vecs.T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def rotation_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def process_pose(pose_dir: str, K: np.ndarray, dist: np.ndarray,
                 rng: np.random.Generator, photo_angle_deg: float,
                 init: tuple[np.ndarray, np.ndarray] | None = None,
                 use_guided: bool = False) -> PoseObs:
    image = cv2.imread(os.path.join(pose_dir, "color.png"))
    if image is None:
        raise ValueError(f"{pose_dir}: color.png not readable")
    points = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))
    camera_plane, image_points, up_cam, center_cam, board_polygon = detect_camera_plane(image, K, dist)
    lidar_plane, inliers = extract_lidar_board(points, rng, pose_dir, camera_plane=camera_plane,
                                               quad=board_polygon, K=K,
                                               dist=dist, init=init, photo_angle=photo_angle_deg,
                                               use_guided=use_guided)
    if photo_angle_deg:
        Rz = rotation_z(photo_angle_deg)
        lidar_plane = Plane(Rz @ lidar_plane.normal, lidar_plane.offset)
        inliers = inliers @ Rz.T
    return PoseObs(os.path.basename(pose_dir.rstrip("/")), pose_dir, camera_plane, lidar_plane,
                   image, image_points, inliers, up_cam, center_cam, np.median(inliers, axis=0), board_polygon)


def draw_validation(obs: PoseObs, R: np.ndarray, t: np.ndarray,
                    K: np.ndarray, dist: np.ndarray) -> str:
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(obs.lidar_inliers, rvec, t, K, dist)
    overlay = obs.image.copy()
    for u, v in projected.reshape(-1, 2):
        if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
            cv2.circle(overlay, (int(u), int(v)), 1, (0, 0, 255), -1)
    for u, v in obs.image_points:
        cv2.circle(overlay, (int(u), int(v)), 3, (0, 255, 0), 1)
    out = os.path.join(obs.pose_dir, "validation_overlay.png")
    cv2.imwrite(out, overlay)
    return out


def solve_all(obses: list[PoseObs]) -> tuple[np.ndarray, np.ndarray]:
    lidar_normals = np.array([o.lidar_plane.normal for o in obses])
    camera_normals = np.array([o.camera_plane.normal for o in obses])
    R = solve_rotation(lidar_normals.T, camera_normals.T)
    centers_l = np.array([o.center_lidar for o in obses])
    centers_c = np.array([o.center_cam for o in obses])
    t = (centers_c - (R @ centers_l.T).T).mean(axis=0)
    return R, t


def normals_residual(obses: list[PoseObs], R: np.ndarray) -> np.ndarray:
    lidar_normals = np.array([o.lidar_plane.normal for o in obses])
    camera_normals = np.array([o.camera_plane.normal for o in obses])
    return np.degrees(np.arccos(np.clip(np.sum((R @ lidar_normals.T).T * camera_normals, axis=1), -1, 1)))


def largest_component_mask(inliers: np.ndarray, normal: np.ndarray,
                           cell: float = 0.02) -> np.ndarray | None:
    """板面 2D 占据图上只保留最大连通域，去掉不连通的碎片（支架/后方架子）。"""
    c = inliers.mean(axis=0)
    cov = (inliers - c).T @ (inliers - c)
    _, vecs = np.linalg.eigh(cov)
    e1, e2 = vecs[:, 1], vecs[:, 2]
    u = (inliers - c) @ e1
    v = (inliers - c) @ e2
    iu = np.floor((u - u.min()) / cell).astype(np.int32)
    iv = np.floor((v - v.min()) / cell).astype(np.int32)
    img = np.zeros((iv.max() + 1, iu.max() + 1), np.uint8)
    img[iv, iu] = 1
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels[iv, iu] == largest


def board_window_mask(inliers: np.ndarray, normal: np.ndarray,
                      width: float = BOARD_WIDTH_M, height: float = BOARD_HEIGHT_M,
                      tol: float = BOARD_TRIM_TOL_M) -> np.ndarray | None:
    """用最小外接矩形精确贴合板面，并按已知板尺寸 (0.60×0.84 m) 校验。

    旧实现是 ±40°/5° 粗搜索 + 直方图上边缘，矩形方向误差可达 2.5°，
    裁出的点集被切歪（PCA 尺寸比真实板大），平面的法向也随之被带偏。
    改为 cv2.minAreaRect 精确求方向，再按已知尺寸裁边，并打印实测尺寸便于核对。
    """
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    v = up - (up @ normal) * normal
    if np.linalg.norm(v) < 1e-6:
        return None
    v /= np.linalg.norm(v)
    u = np.cross(normal, v)
    d = inliers - inliers.mean(axis=0)
    uv = np.column_stack([d @ u, d @ v]).astype(np.float32)

    keep = np.ones(len(uv), bool)
    for it in range(3):
        if int(keep.sum()) < 200:
            return None
        (cx, cy), (w, h), ang = cv2.minAreaRect(uv[keep])
        long_s, short_s = max(float(w), float(h)), min(float(w), float(h))
        th = np.radians(ang)
        ca, sa = np.cos(th), np.sin(th)
        du, dv = uv[:, 0] - cx, uv[:, 1] - cy
        ru = du * ca + dv * sa
        rv = -du * sa + dv * ca
        if h >= w:
            rlong, rshort = rv, ru
        else:
            rlong, rshort = ru, rv
        keep = (np.abs(rshort) < width / 2.0 + tol) & (np.abs(rlong) < height / 2.0 + tol)
        ok = (0.5 * height < long_s < 1.5 * height) and (0.5 * width < short_s < 1.5 * width)
        print(f"    [board] 迭代{it}: 最小外接矩形 {long_s:.2f}×{short_s:.2f} m "
              f"(期望 {height:.2f}×{width:.2f}) {'OK' if ok else '⚠ 尺寸不符'} → 保留 {int(keep.sum())} 点")
    return keep


def refine_pose_plane(obs: PoseObs, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                      dist: np.ndarray, margin_px: float = 12.0) -> PoseObs:
    # 板已由 segment_board 按已知尺寸直接分割, 不再用相机投影/旧窗口重裁剪(避免先入为主)
    return obs
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(obs.lidar_inliers, rvec, t, K, dist)
    projected = projected.reshape(-1, 2)
    polygon = obs.board_polygon.astype(np.float32)
    keep = np.array([cv2.pointPolygonTest(polygon, (float(u), float(v)), True) >= -margin_px
                     for u, v in projected])
    kept = obs.lidar_inliers[keep]
    if len(kept) >= 200:
        rough = fit_plane_svd(kept)
        comp = largest_component_mask(kept, rough.normal)
        if comp is not None and int(comp.sum()) >= 200:
            kept = kept[comp]
        window = board_window_mask(kept, rough.normal)
        if window is not None and int(window.sum()) >= 200:
            kept = kept[window]
    write_ply_xyz(os.path.join(obs.pose_dir, "dbg_4_board_selected.ply"), kept)
    if len(kept) < 50:
        print(f"  {obs.name}: camera selection kept too few points ({len(kept)}); keeping raw plane")
        return obs
    return replace(obs, lidar_plane=fit_plane_svd(kept), lidar_inliers=kept,
                   center_lidar=np.median(kept, axis=0))


def image_space_residuals(obses: list[PoseObs], R: np.ndarray, t: np.ndarray,
                          K: np.ndarray, dist: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    rvec = cv2.Rodrigues(R)[0]
    out = []
    for o in obses:
        proj, _ = cv2.projectPoints(o.lidar_inliers, rvec, t, K, dist)
        proj = proj.reshape(-1, 2)
        pc = (R @ o.lidar_inliers.T).T + t
        ok = np.isfinite(proj).all(axis=1) & (pc[:, 2] > 0.2)
        if int(ok.sum()) < 50:
            continue
        cc, _ = cv2.projectPoints(o.center_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
        out.append((o.name, np.median(proj[ok], axis=0) - cc.reshape(2), float(np.median(pc[ok, 2]))))
    return out


def refine_image_space(obses: list[PoseObs], R: np.ndarray, t: np.ndarray,
                       K: np.ndarray, dist: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """在图像空间收敛：按 fy·Δy/Z ≈ Δv 的关系修正相机系平移，使板投影落到图像棋盘格中心。"""
    for _ in range(iters):
        res = image_space_residuals(obses, R, t, K, dist)
        if not res:
            break
        dx = -float(np.mean([du * Z / K[0, 0] for _, (du, _), Z in res]))
        dy = -float(np.mean([dv * Z / K[1, 1] for _, (_, dv), Z in res]))
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            break
        t = t + np.array([dx, dy, 0.0])
    return R, t


def report_and_solve(obses: list[PoseObs], K: np.ndarray, dist: np.ndarray, out_path: str) -> int:
    print(f"poses: {len(obses)}")
    for obs in obses:
        print(f"  {obs.name}: n_c={obs.camera_plane.normal.round(4).tolist()} d_c={obs.camera_plane.offset:.4f} "
              f"| n_l={obs.lidar_plane.normal.round(4).tolist()} d_l={obs.lidar_plane.offset:.4f} "
              f"| inliers={len(obs.lidar_inliers)}")

    R, t = solve_all(obses)
    refined = [refine_pose_plane(o, R, t, K, dist) for o in obses]
    R, t = solve_all(refined)
    err = normals_residual(refined, R)
    print(f"[refined] normal angle error (deg): {err.round(3).tolist()} rms={np.sqrt((err**2).mean()):.3f}")
    for o in refined:
        overlap = quad_overlap_score(o.lidar_inliers, o.board_polygon, K, dist, R, t, 0.0)
        print(f"  {o.name}: 板面落格率={overlap:.0%} (用解出外参投影，越高越好)")
    centers_l = np.array([o.center_lidar for o in refined])
    centers_c = np.array([o.center_cam for o in refined])
    t_diff = centers_c - (R @ centers_l.T).T
    print(f"[refined] board-center residual (m): {np.linalg.norm(t_diff - t_diff.mean(axis=0), axis=1).round(4).tolist()}")

    res0 = image_space_residuals(refined, R, t, K, dist)
    if res0:
        du0 = np.array([r[1] for r in res0])
        print(f"[image]   before: Δu mean={du0[:,0].mean():+.1f}px  Δv mean={du0[:,1].mean():+.1f}px  "
              f"|Δ| mean={np.linalg.norm(du0, axis=1).mean():.1f}px")
    R, t = refine_image_space(refined, R, t, K, dist)
    res1 = image_space_residuals(refined, R, t, K, dist)
    if res1:
        du1 = np.array([r[1] for r in res1])
        print(f"[image]   after : Δu mean={du1[:,0].mean():+.1f}px  Δv mean={du1[:,1].mean():+.1f}px  "
              f"|Δ| mean={np.linalg.norm(du1, axis=1).mean():.1f}px")
        for name, (du, dv), _ in res1:
            print(f"    {name}: Δu={du:+.1f}px Δv={dv:+.1f}px")
    print(f"diagnostic: R@merged_up(Z)={(R @ np.array([0.0, 0.0, 1.0])).round(3).tolist()} (camera up ≈ [0,-1,0])")

    for obs in refined:
        print(f"validation overlay: {draw_validation(obs, R, t, K, dist)}")

    data = {
        "lidar_to_camera": {
            "rotation_matrix": R.flatten().tolist(),
            "translation": t.tolist(),
        },
        "note": "p_camera = R @ p_lidar + t (merged-frame LiDAR; see AGENTS.md)",
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"saved {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera-LiDAR checkerboard calibration")
    parser.add_argument("pose_dirs", nargs="+", help="pose dirs containing color.png + merged.ply")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--output", default="config/camera_extrinsics.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--photo-angle", type=float, default=90.0,
                        help="turntable angle of the photos (deg); merged.ply is referenced to 0 deg, "
                             "so n_l is rotated by this to the photo frame. Default 90 (ready).")
    parser.add_argument("--guided", action="store_true",
                        help="Use camera-guided plane fitting instead of the RANSAC size-gate search.")
    parser.add_argument("--init-extrinsics", default="config/camera_extrinsics.yaml",
                        help="Initial extrinsics used to pick the board plane by projecting candidates into the "
                             "photo's checkerboard quad (falls back to the size gate if the file is missing).")
    args = parser.parse_args()

    K, dist = load_camera_info(args.camera_info)
    init = None
    if args.init_extrinsics and os.path.exists(args.init_extrinsics):
        ex = yaml.safe_load(open(args.init_extrinsics))["lidar_to_camera"]
        init = (np.array(ex["rotation_matrix"]).reshape(3, 3), np.array(ex["translation"]))
        print(f"using init extrinsics: {args.init_extrinsics}")
    rng = np.random.default_rng(args.seed)
    obses = []
    for pose_dir in args.pose_dirs:
        try:
            obs = process_pose(pose_dir, K, dist, rng, args.photo_angle, init=init,
                               use_guided=args.guided)
        except (ValueError, OSError) as exc:
            print(f"skip {pose_dir}: {exc}", file=sys.stderr)
            continue
        obses.append(obs)
    if not obses:
        print("no usable poses", file=sys.stderr)
        return 2
    return report_and_solve(obses, K, dist, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
