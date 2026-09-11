#!/usr/bin/env python3
"""Camera-LiDAR extrinsic calibration from a planar checkerboard.

Each pose directory must contain `color.png` and `merged.ply`. See AGENTS.md for the
coordinate frame and ROI conventions.

Pipeline per pose:
    image  -> findChessboardCorners(4x6) -> solvePnP -> camera plane (n_c, d_c)
    cloud  -> ROI crop -> height filter -> RANSAC largest plane -> LiDAR plane (n_l, d_l)
Then:  n_c = R n_l   and   d_c - d_l = n_c · t   solved over all poses.

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
ROI_MAX_DIST_M = 3.0
ROI_HALF_ANGLE_DEG = 20.0
HEIGHT_MIN_M = 0.4
HEIGHT_MAX_M = 2.6
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
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
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


def crop_roi(points: np.ndarray, max_dist: float, half_angle_deg: float) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1)
    theta = np.degrees(np.arctan2(points[:, 1], -points[:, 0]))
    keep = (radius < max_dist) & (np.abs(theta) < half_angle_deg)
    return points[keep]


def detect_ground_z(points: np.ndarray, bins: int = 200) -> float:
    counts, edges = np.histogram(points[:, 2], bins=bins)
    peak = int(np.argmax(counts))
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def extract_lidar_board(points: np.ndarray, rng: np.random.Generator,
                        out_dir: str) -> tuple[Plane, np.ndarray]:
    roi = crop_roi(points, ROI_MAX_DIST_M, ROI_HALF_ANGLE_DEG)
    write_ply_xyz(os.path.join(out_dir, "dbg_1_roi.ply"), roi)
    ground_z = detect_ground_z(roi)
    height = roi[:, 2] - ground_z
    band = roi[(height > HEIGHT_MIN_M) & (height < HEIGHT_MAX_M)]
    write_ply_xyz(os.path.join(out_dir, "dbg_2_height.ply"), band)
    if len(band) < 100:
        raise ValueError("too few points after ROI/height filtering")
    plane = ransac_plane(band, rng)
    inliers = band[np.abs(band @ plane.normal - plane.offset) < PLANE_INLIER_M]
    write_ply_xyz(os.path.join(out_dir, "dbg_3_plane.ply"), inliers)
    return plane, inliers


def find_board_corners(gray: np.ndarray) -> np.ndarray:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags=flags)
    if found:
        return corners
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    found, corners = cv2.findChessboardCornersSB(thresh, (BOARD_COLS, BOARD_ROWS))
    if found:
        return corners
    raise ValueError(f"checkerboard ({BOARD_COLS}x{BOARD_ROWS} inner corners) not found")


def detect_camera_plane(image: np.ndarray, K: np.ndarray, dist: np.ndarray
                        ) -> tuple[Plane, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(
        gray, find_board_corners(gray), (5, 5), (-1, -1),
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
        [-0.5 * SQUARE_SIZE_M, -0.5 * SQUARE_SIZE_M, 0.0],
        [(BOARD_COLS - 0.5) * SQUARE_SIZE_M, -0.5 * SQUARE_SIZE_M, 0.0],
        [(BOARD_COLS - 0.5) * SQUARE_SIZE_M, (BOARD_ROWS - 0.5) * SQUARE_SIZE_M, 0.0],
        [-0.5 * SQUARE_SIZE_M, (BOARD_ROWS - 0.5) * SQUARE_SIZE_M, 0.0],
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
                   image, image_points, inliers, up_cam, center_cam, inliers.mean(axis=0), board_polygon)


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


def refine_pose_plane(obs: PoseObs, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                      dist: np.ndarray, margin_px: float = 12.0) -> PoseObs:
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(obs.lidar_inliers, rvec, t, K, dist)
    projected = projected.reshape(-1, 2)
    polygon = obs.board_polygon.astype(np.float32)
    keep = np.array([cv2.pointPolygonTest(polygon, (float(u), float(v)), True) >= -margin_px
                     for u, v in projected])
    kept = obs.lidar_inliers[keep]
    write_ply_xyz(os.path.join(obs.pose_dir, "dbg_4_board_selected.ply"), kept)
    if len(kept) < 50:
        print(f"  {obs.name}: camera selection kept too few points ({len(kept)}); keeping raw plane")
        return obs
    return replace(obs, lidar_plane=fit_plane_svd(kept), lidar_inliers=kept,
                   center_lidar=kept.mean(axis=0))


def report_and_solve(obses: list[PoseObs], K: np.ndarray, dist: np.ndarray, out_path: str) -> int:
    print(f"poses: {len(obses)}")
    for obs in obses:
        print(f"  {obs.name}: n_c={obs.camera_plane.normal.round(4).tolist()} d_c={obs.camera_plane.offset:.4f} "
              f"| n_l={obs.lidar_plane.normal.round(4).tolist()} d_l={obs.lidar_plane.offset:.4f} "
              f"| inliers={len(obs.lidar_inliers)}")

    R, t = solve_all(obses)
    raw_err = normals_residual(obses, R)
    print(f"[raw]     normal angle error (deg): {raw_err.round(3).tolist()} rms={np.sqrt((raw_err**2).mean()):.3f}")

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
