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
import os
import sys
from dataclasses import dataclass, replace

import cv2
import numpy as np
import yaml

SQUARE_SIZE_M = 0.12
BOARD_COLS = 4
BOARD_ROWS = 6
BOARD_WIDTH_M = 0.60
BOARD_HEIGHT_M = 0.84
BOARD_TRIM_TOL_M = 0.05
GROUND_MAX_TILT_COS = 0.96
GROUND_MIN_SPAN_M = 1.2
GROUND_SLAB_M = 0.05
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


def search_board_plane(points: np.ndarray, rng: np.random.Generator,
                       rounds: int = 8) -> tuple[Plane, np.ndarray]:
    rest = points
    for _ in range(rounds):
        if len(rest) < 500:
            break
        plane = ransac_plane(rest, rng)
        inliers = rest[np.abs(rest @ plane.normal - plane.offset) < PLANE_INLIER_M]
        if _is_board_plane(inliers, plane.normal):
            return plane, inliers
        keep = np.abs(rest @ plane.normal - plane.offset) >= PLANE_INLIER_M
        if keep.sum() < 500:
            break
        rest = rest[keep]
    raise ValueError("no board-like vertical plane found")


def extract_lidar_board(points: np.ndarray, rng: np.random.Generator,
                        out_dir: str) -> tuple[Plane, np.ndarray]:
    ground = find_ground_plane(points, rng)
    radius = np.linalg.norm(points, axis=1)
    scene = points[(np.abs(points @ ground.normal - ground.offset) > GROUND_SLAB_M)
                   & (radius > SELF_RETURN_RADIUS_M)]
    write_ply_xyz(os.path.join(out_dir, "dbg_2_no_ground.ply"), scene)
    plane, inliers = search_board_plane(scene, rng)
    rough = fit_plane_svd(inliers)
    comp = largest_component_mask(inliers, rough.normal)
    if comp is not None and int(comp.sum()) >= 200:
        inliers = inliers[comp]
        plane = fit_plane_svd(inliers)
    write_ply_xyz(os.path.join(out_dir, "dbg_3_plane.ply"), inliers)
    return plane, inliers


def find_board_corners(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns = [(BOARD_COLS, BOARD_ROWS), (BOARD_ROWS, BOARD_COLS)]
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
        cv2.threshold(image[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]
    for th in thresh_variants:
        for pat in patterns:
            found, corners = cv2.findChessboardCornersSB(th, pat)
            if found:
                return corners
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for pat in patterns:
        found, corners = cv2.findChessboardCorners(up, pat, flags)
        if found:
            return corners / 2.0
    raise ValueError(f"checkerboard ({BOARD_COLS}x{BOARD_ROWS} inner corners) not found")


def detect_camera_plane(image: np.ndarray, K: np.ndarray, dist: np.ndarray
                        ) -> tuple[Plane, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(
        gray, find_board_corners(image), (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    object_points = board_object_points()
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
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
                 rng: np.random.Generator, photo_angle_deg: float) -> PoseObs:
    image = cv2.imread(os.path.join(pose_dir, "color.png"))
    if image is None:
        raise ValueError(f"{pose_dir}: color.png not readable")
    points = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))
    camera_plane, image_points, up_cam, center_cam, board_polygon = detect_camera_plane(image, K, dist)
    lidar_plane, inliers = extract_lidar_board(points, rng, pose_dir)
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
    """按已知板尺寸的矩形窗口裁掉板外的直线延展（支架/后方架子）；上边缘由密度剖面确定。"""
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    v_base = up - (up @ normal) * normal
    norm = np.linalg.norm(v_base)
    if norm < 1e-6:
        return None
    v_base /= norm
    u_base = np.cross(normal, v_base)
    d = inliers - inliers.mean(axis=0)
    best_keep, best_count = None, -1
    for angle in np.arange(-40.0, 40.1, 5.0):
        rad = np.radians(angle)
        u_dir = np.cos(rad) * u_base + np.sin(rad) * v_base
        v_dir = -np.sin(rad) * u_base + np.cos(rad) * v_base
        u = d @ u_dir
        v = d @ v_dir
        edges = np.arange(v.min(), v.max() + 0.02, 0.02)
        if len(edges) < 4:
            continue
        hist, _ = np.histogram(v, bins=edges)
        dense = hist > 0.3 * hist.max()
        peak = int(np.argmax(hist))
        top = peak
        while top < len(dense) - 1 and dense[top + 1]:
            top += 1
        bottom = peak
        while bottom > 0 and dense[bottom - 1]:
            bottom -= 1
        v_top = float(edges[top + 1])
        v_body = v[(v > float(edges[bottom])) & (v < v_top)]
        if len(v_body) < 100:
            continue
        u_mid = float(np.median(u[(v > v_top - height) & (v < v_top)]))
        keep = ((np.abs(u - u_mid) < width / 2.0 + tol)
                & (v > v_top - height - tol) & (v < v_top + tol))
        if int(keep.sum()) > best_count:
            best_count, best_keep = int(keep.sum()), keep
    return best_keep


def refine_pose_plane(obs: PoseObs, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                      dist: np.ndarray, margin_px: float = 12.0) -> PoseObs:
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
    centers_l = np.array([o.center_lidar for o in refined])
    centers_c = np.array([o.center_cam for o in refined])
    t_diff = centers_c - (R @ centers_l.T).T
    print(f"[refined] board-center residual (m): {np.linalg.norm(t_diff - t_diff.mean(axis=0), axis=1).round(4).tolist()}")
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
    args = parser.parse_args()

    K, dist = load_camera_info(args.camera_info)
    rng = np.random.default_rng(args.seed)
    obses = []
    for pose_dir in args.pose_dirs:
        try:
            obs = process_pose(pose_dir, K, dist, rng, args.photo_angle)
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
