#!/usr/bin/env python3
"""Densify a merged LiDAR cloud into a per-pixel RGBD cloud (LiDAR->RGBD D2C).

Pipeline (mirrors the Orbbec SDK software D2C in reverse — building the aligned depth
map from the sparse LiDAR cloud instead of from a dense depth frame):

  1. Project LiDAR points into the color camera frame (Rz(photo_angle) then (R, t), K, dist);
  2. Splat into a color-image-sized depth buffer, min-z wins on conflicts (occlusion),
     optional 2x2 dilation like the SDK gap-fill;
  3. Fill remaining holes from the nearest valid pixel (distance-transform labels),
     limited by --max-fill-px so large voids stay empty instead of inventing geometry;
  4. Unproject every valid pixel through K^-1 (undistorted ray) x depth -> dense cloud,
     colored by its own pixel.

Frame handling and CLI conventions follow colorize_pointcloud.py.

Usage:
    python3 dense_colorize_pointcloud.py POSE_DIR [--extrinsics config/camera_extrinsics.yaml]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from colorize_pointcloud import (load_camera_info, load_extrinsics, read_ply_xyz,
                                 rotation_z, write_ply_rgb)

INF = np.inf


def splat_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray, shape: tuple[int, int],
                dilate: bool = True) -> np.ndarray:
    """Splat points into an HxW depth buffer; nearest z wins per pixel.

    dilate=True also writes the z-min into the right/bottom/bottom-right neighbors
    (SDK-style gap fill) so each return covers up to 2x2 pixels.
    """
    h, w = shape
    depth = np.full((h, w), INF, dtype=np.float32)
    ui = np.floor(u + 0.5).astype(int)
    vi = np.floor(v + 0.5).astype(int)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)) if dilate else ((0, 0),):
        uu, vv = ui + du, vi + dv
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        idx = (vv[ok] * w + uu[ok]).astype(np.int64)
        np.minimum.at(depth.reshape(-1), idx, z[ok])
    return depth


def fill_holes_nearest(depth: np.ndarray, max_fill_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill inf holes from the nearest valid pixel; returns (filled, valid) masks applied."""
    valid = np.isfinite(depth)
    if valid.all() or not valid.any():
        return depth, valid
    # distanceTransform: zero pixels = valid sources, others get distance + nearest-source label
    dist, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8),
                                                   cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    src = depth.reshape(-1)
    filled = depth.copy()
    fillable = ~valid & (dist <= max_fill_px)
    # DIST_LABEL_PIXEL 标签 = 有效像素按行优先序的 1 起始序号(非线性索引), 需二次映射
    zero_idx = np.flatnonzero(valid.reshape(-1))
    filled[fillable] = src[zero_idx[labels[fillable] - 1]]
    return filled, np.isfinite(filled)


def unproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
              dist: np.ndarray) -> np.ndarray:
    """Back-project valid depth pixels to camera-frame 3D points (z = depth value)."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    uv = np.column_stack([us[valid], vs[valid]]).reshape(-1, 1, 2)
    rays2d = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)
    rays = np.column_stack([rays2d, np.ones(len(rays2d))])
    z = depth[valid].astype(np.float64)
    return rays * (z / rays[:, 2])[:, None]


def check_bijection(pts_cam: np.ndarray, src_uv: np.ndarray, K: np.ndarray,
                    dist: np.ndarray) -> dict:
    """Reproject the dense cloud back to the image and verify a strict pixel<->point bijection.

    Checks (1) every point lands back on its source pixel (sub-pixel reprojection error)
    and (2) no two points share a pixel.
    """
    proj, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    proj = proj.reshape(-1, 2)
    offset = float(np.abs(proj - src_uv).max()) if len(proj) else 0.0
    h, w = int(src_uv[:, 1].max()) + 1, int(src_uv[:, 0].max()) + 1
    pix_idx = np.round(proj).astype(np.int64)[:, 1] * w + np.round(proj).astype(np.int64)[:, 0]
    unique_pixels = int(np.unique(pix_idx).size)
    ok = (unique_pixels == len(pts_cam)) and (offset < 0.5) and \
         (np.abs(proj - np.round(proj)).max() < 0.5 if len(proj) else True)
    return {"ok": bool(ok), "unique_pixels": unique_pixels, "n_points": len(pts_cam),
            "max_offset_px": offset}


def dense_colorize(pose_dir: str, extrinsics_path: str, camera_info_path: str,
                   photo_angle: float, dilate: bool = True, max_fill_px: float = 3.0,
                   min_dist: float | None = None, max_dist: float | None = None,
                   output: str = "dense_colored.ply",
                   save_index_map: bool = False) -> tuple[str, int, float, dict]:
    """Build a dense RGBD cloud from merged.ply + color.png.

    min_dist/max_dist 为 None 时不做距离裁剪(量程由调用方决定: GUI 传实时 Range,
    独立运行默认不裁剪, 远点只受相机视场约束)。相机视场外的点无法溅射, 天然不进 dense。
    Returns (path, n_points, coverage, bijection_check). The check reprojects the output
    back to the image and confirms each point sits on its own source pixel exactly 1:1."""
    K, dist = load_camera_info(camera_info_path)
    R, t = load_extrinsics(extrinsics_path)
    image = cv2.imread(os.path.join(pose_dir, "color.png"))
    if image is None:
        raise ValueError(f"{pose_dir}: color.png not readable")
    points = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))

    r = np.linalg.norm(points, axis=1)
    keep = np.ones(len(points), dtype=bool)
    if min_dist is not None:
        keep &= r > min_dist
    if max_dist is not None:
        keep &= r <= max_dist
    points = points[keep]

    R_full = R @ rotation_z(photo_angle)
    rvec = cv2.Rodrigues(R_full)[0]
    projected, _ = cv2.projectPoints(points, rvec, t, K, dist)
    projected = projected.reshape(-1, 2)
    cam = (R_full @ points.T).T + t
    z = cam[:, 2]

    h, w = image.shape[:2]
    finite = np.isfinite(projected).all(axis=1) & (np.abs(projected) < 1.0e6).all(axis=1)
    vis = finite & (z > 0) & (projected[:, 0] >= 0) & (projected[:, 0] < w) \
        & (projected[:, 1] >= 0) & (projected[:, 1] < h)

    depth = splat_depth(projected[vis, 0], projected[vis, 1], z[vis], (h, w), dilate=dilate)
    filled, valid = fill_holes_nearest(depth, max_fill_px)

    pts_cam = unproject(filled, valid, K, dist)
    us_grid, vs_grid = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src_uv = np.column_stack([us_grid[valid], vs_grid[valid]])
    check = check_bijection(pts_cam, src_uv, K, dist)
    R_wc = R_full.T  # world(photo-frame lidar) = R^T * cam
    pts_world = (R_wc @ pts_cam.T).T - (R_wc @ t)
    colors = image[valid][:, ::-1]

    out_path = os.path.join(pose_dir, output)
    write_ply_rgb(out_path, pts_world, colors)
    if save_index_map:
        # HxW int32: 有效像素 → PLY 行号(光栅序), 无效 → -1; 供图像 mask(YOLO seg 等)直接查点云区域
        index_map = np.full((h, w), -1, dtype=np.int32)
        index_map[valid] = np.arange(len(pts_world), dtype=np.int32)
        np.save(os.path.join(pose_dir, os.path.splitext(output)[0] + "_index.npy"), index_map)
    return out_path, len(pts_world), float(valid.mean()), check


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Densify merged.ply into a per-pixel RGBD cloud (LiDAR->RGBD D2C)")
    parser.add_argument("pose_dir", help="folder with color.png + merged.ply")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--no-dilate", action="store_true", help="关闭2x2膨胀溅射")
    parser.add_argument("--max-fill-px", type=float, default=3.0,
                        help="空洞最近邻填补的像素半径上限(超出保持无效, 不虚构几何)")
    parser.add_argument("--min-dist", type=float, default=None,
                        help="距离下限(米), 默认不裁剪; 调用方应按扫描时 GUI Range 传入")
    parser.add_argument("--max-dist", type=float, default=None,
                        help="距离上限(米), 默认不裁剪; 调用方应按扫描时 GUI Range 传入")
    parser.add_argument("--output", default="dense_colored.ply")
    parser.add_argument("--save-index-map", action="store_true",
                        help="额外输出 <output去扩展名>_index.npy (HxW int32, 像素→PLY行号, -1=无点)")
    args = parser.parse_args()

    out, n, cov, check = dense_colorize(args.pose_dir, args.extrinsics, args.camera_info,
                                        args.photo_angle, dilate=not args.no_dilate,
                                        max_fill_px=args.max_fill_px, min_dist=args.min_dist,
                                        max_dist=args.max_dist, output=args.output,
                                        save_index_map=args.save_index_map)
    verdict = ("PASS" if check["ok"] else "FAIL")
    print(f"{args.pose_dir}: {n} dense points, pixel coverage {cov:.1%}, "
          f"1:1像素↔点校验{verdict} (唯一像素{check['unique_pixels']}/{check['n_points']}, "
          f"最大回投偏差{check['max_offset_px']:.4f}px) -> {out}")
    return 0 if check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
