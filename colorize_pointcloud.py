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
    return points[:, :3]


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


def colorize(pose_dir: str, extrinsics_path: str, camera_info_path: str,
             photo_angle: float, output: str = "colored.ply") -> tuple[str, int, int]:
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
    safe = np.where(good[:, None], projected, 0.0)
    u = np.round(safe[:, 0]).astype(int)
    v = np.round(safe[:, 1]).astype(int)
    visible = good & (cam_z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    colors = np.full((len(points), 3), 30, dtype=np.uint8)
    colors[visible] = image[v[visible], u[visible]][:, ::-1]

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
    args = parser.parse_args()

    out, total, colored = colorize(args.pose_dir, args.extrinsics, args.camera_info,
                                   args.photo_angle, args.output)
    print(f"{args.pose_dir}: {total} points, {colored} colored ({colored / max(total, 1):.1%}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
