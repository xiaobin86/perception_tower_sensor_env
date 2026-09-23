#!/usr/bin/env python3
"""Colorize a merged point cloud by projecting it onto its camera image.

Frame handling: merged.ply is referenced to turntable angle 0, while the camera photo is at
`--photo-angle`; the saved extrinsic maps the *photo-frame* LiDAR to the camera, so the raw cloud
is rotated by Rz(photo_angle) before applying (R, t).

Usage:
    python3 colorize_pointcloud.py POSE_DIR [--extrinsics config/camera_extrinsics.yaml]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import yaml


def read_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for header_lines, line in enumerate(f, start=1):
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path}: no 'end_header'")
    points = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    # OpenCV 5.x projectPoints rejects non-contiguous input (loadtxt slice is a view)
    return np.ascontiguousarray(points[:, :3])


def write_ply_rgb(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    data = np.column_stack([points, colors.astype(np.float64)])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
    with open(path, "a") as f:
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %.0f %.0f %.0f")


def rotation_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_camera_info(path: str) -> tuple[np.ndarray, np.ndarray]:
    info = yaml.safe_load(open(path))
    return np.array(info["k"]).reshape(3, 3), np.array(info["d"]).reshape(-1, 1)


def load_extrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    ex = yaml.safe_load(open(path))["lidar_to_camera"]
    return np.array(ex["rotation_matrix"]).reshape(3, 3), np.array(ex["translation"])


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear RGB sampling at continuous (u, v).

    Out-of-image positions are clamped to the border (matches the "visible" mask semantics
    in colorize, which already drops points outside the frame; clamping only affects the
    fractional edge case).
    """
    h, w = image.shape[:2]
    u = np.clip(u, 0.0, w - 1.0)
    v = np.clip(v, 0.0, h - 1.0)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    du = (u - u0.astype(np.float64))[:, None]
    dv = (v - v0.astype(np.float64))[:, None]
    img = image.astype(np.float64)
    top = img[v0, u0] * (1.0 - du) + img[v0, u1] * du
    bot = img[v1, u0] * (1.0 - du) + img[v1, u1] * du
    return np.round(top * (1.0 - dv) + bot * dv).astype(np.uint8)[:, ::-1]  # BGR -> RGB


def sample_colors(image: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray,
                  sampling: str = "nearest") -> np.ndarray:
    """Sample RGB colors for points at continuous pixel coords (u, v).

    mask=False positions keep the default gray (30, 30, 30).
    sampling: "nearest" (round to pixel, legacy behavior) or "bilinear" (sub-pixel interp).
    """
    if sampling not in ("nearest", "bilinear"):
        raise ValueError(f"sampling must be 'nearest' or 'bilinear', got {sampling!r}")
    colors = np.full((len(u), 3), 30, dtype=np.uint8)
    if not mask.any():
        return colors
    h, w = image.shape[:2]
    if sampling == "nearest":
        un = np.round(u[mask]).astype(int)
        vn = np.round(v[mask]).astype(int)
        colors[mask] = image[vn, un][:, ::-1]
    else:
        colors[mask] = bilinear_sample(image, u[mask], v[mask])
    return colors


def colorize(pose_dir: str, extrinsics_path: str, camera_info_path: str,
             photo_angle: float, output: str = "colored.ply",
             sampling: str = "bilinear") -> tuple[str, int, int]:
    """Project merged.ply onto color.png and write an RGB PLY. Returns (path, total, colored)."""
    K, dist = load_camera_info(camera_info_path)
    R, t = load_extrinsics(extrinsics_path)
    image = cv2.imread(os.path.join(pose_dir, "color.png"))
    if image is None:
        raise ValueError(f"{pose_dir}: color.png not readable")
    points = read_ply_xyz(os.path.join(pose_dir, "merged.ply"))

    R_full = R @ rotation_z(photo_angle)
    rvec = cv2.Rodrigues(R_full)[0]
    projected, _ = cv2.projectPoints(points, rvec, t, K, dist)
    projected = projected.reshape(-1, 2)
    cam_z = (R_full @ points.T + t.reshape(3, 1)).T[:, 2]

    h, w = image.shape[:2]
    finite = np.isfinite(projected) & (np.abs(projected) < 1.0e6)
    good = finite[:, 0] & finite[:, 1]
    visible = good & (cam_z > 0) & (projected[:, 0] >= 0) & (projected[:, 0] < w) \
        & (projected[:, 1] >= 0) & (projected[:, 1] < h)

    colors = sample_colors(image, projected[:, 0], projected[:, 1], visible, sampling)

    out_path = os.path.join(pose_dir, output)
    write_ply_rgb(out_path, points, colors)
    return out_path, len(points), int(visible.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Colorize merged point cloud from camera image")
    parser.add_argument("pose_dir", help="folder with color.png + merged.ply")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--output", default="colored.ply")
    parser.add_argument("--sampling", choices=["bilinear", "nearest"], default="bilinear",
                        help="bilinear=亚像素双线性插值(默认, 边缘更平滑); nearest=整像素最近邻(旧行为)")
    args = parser.parse_args()

    out, total, colored = colorize(args.pose_dir, args.extrinsics, args.camera_info,
                                   args.photo_angle, args.output, sampling=args.sampling)
    print(f"{args.pose_dir}: {total} points, {colored} colored ({colored / max(total, 1):.1%}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
