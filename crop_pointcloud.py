#!/usr/bin/env python3
"""Crop a merged turntable point cloud to a distance range and a horizontal wedge.

Coordinate conventions (see AGENTS.md):
    Z  = vertical (turntable rotation axis)
    XY = horizontal plane
    horizontal angle = atan2(y, -x)   (0 = -X direction, positive toward +Y)

Usage:
    python3 crop_pointcloud.py merged.ply cropped.ply
    python3 crop_pointcloud.py merged.ply cropped.ply --max-dist 3 --half-angle 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def read_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for header_lines, line in enumerate(f, start=1):
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path}: no 'end_header' found (not an ASCII PLY?)")
    points = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path}: expected Nx3 vertex data")
    return points[:, :3]


def write_ply_xyz(path: str, points: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f")


def horizontal_angle_deg(points: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(points[:, 1], -points[:, 0]))


def wrap180(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def crop(points: np.ndarray, max_dist: float, half_angle_deg: float,
         center_deg: float, min_dist: float) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1)
    delta = wrap180(horizontal_angle_deg(points) - center_deg)
    keep = (radius >= min_dist) & (radius < max_dist) & (np.abs(delta) < half_angle_deg)
    return points[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop merged point cloud by distance + horizontal wedge")
    parser.add_argument("input", help="input PLY (ASCII, xyz)")
    parser.add_argument("output", help="output PLY")
    parser.add_argument("--max-dist", type=float, default=3.0, help="max radius (m), default 3.0")
    parser.add_argument("--min-dist", type=float, default=0.0, help="min radius (m), default 0.0")
    parser.add_argument("--half-angle", type=float, default=20.0, help="half wedge angle (deg), default 20")
    parser.add_argument("--center-deg", type=float, default=0.0,
                        help="wedge center in atan2(y,-x) convention; 0 = -X (default)")
    args = parser.parse_args()

    points = read_ply_xyz(args.input)
    kept = crop(points, args.max_dist, args.half_angle, args.center_deg, args.min_dist)
    if len(kept) == 0:
        print("warning: crop removed all points", file=sys.stderr)
    write_ply_xyz(args.output, kept)

    ratio = len(kept) / len(points) if len(points) else 0.0
    print(f"{args.input}: {len(points)} points")
    print(f"{args.output}: {len(kept)} points ({ratio:.1%})")
    if len(kept):
        print(f"  bbox min {kept.min(axis=0).round(3).tolist()} max {kept.max(axis=0).round(3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
