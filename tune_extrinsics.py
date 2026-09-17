#!/usr/bin/env python3
"""从当前外参出发的交互式微调工具：实时把点云投影叠加到照片上，键盘微调并写回。

用法:
    python3 tune_extrinsics.py turntable_output/20260916_081807

按键:
    q/a : tx ∓     w/s : ty ∓     e/d : tz ∓        (单位 mm)
    r/f : rx ∓     t/g : ry ∓     y/h : rz ∓        (单位 deg)
    z/x : 调小/调大平移步长(mm)   c/v : 调小/调大旋转步长(deg)
    p   : 打印当前外参      S : 保存到 config/camera_extrinsics.yaml（先备份 .bak）
    ESC : 退出

提示: 1mm 平移 ≈ 0.37px@1.65m、0.19px@3.5m；0.1° 偏航 ≈ 3mm@1.7m、6mm@3.5m。远处偏右就用 y/h 微调 rz。
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np
import yaml


def rot_axis(axis: str, deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_ply_xyz(path: str) -> np.ndarray:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                break
    return np.loadtxt(path, skiprows=i, dtype=np.float64)[:, :3]


def read_camera_info(path: str) -> tuple[np.ndarray, np.ndarray]:
    info = yaml.safe_load(open(path))
    return np.array(info["k"]).reshape(3, 3), np.array(info["d"]).reshape(-1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive extrinsics fine tuning")
    parser.add_argument("pose_dir")
    parser.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    parser.add_argument("--camera-info", default="config/camera_info.yaml")
    parser.add_argument("--photo-angle", type=float, default=90.0)
    parser.add_argument("--max-points", type=int, default=150000)
    args = parser.parse_args()

    K, dist = read_camera_info(args.camera_info)
    ex = yaml.safe_load(open(args.extrinsics))["lidar_to_camera"]
    R = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t = np.array(ex["translation"])

    image = cv2.imread(os.path.join(args.pose_dir, "color.png"))
    if image is None:
        raise SystemExit(f"{args.pose_dir}/color.png not readable")
    points = load_ply_xyz(os.path.join(args.pose_dir, "merged.ply"))
    if len(points) > args.max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), args.max_points, replace=False)]

    win = "Extrinsics tuning"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    d_t, d_r = 5.0, 0.10

    while True:
        rvec = cv2.Rodrigues(R @ rot_axis("z", args.photo_angle))[0]
        proj, _ = cv2.projectPoints(points, rvec, t, K, dist)
        proj = proj.reshape(-1, 2)
        pc = (R @ ((points @ rot_axis("z", args.photo_angle).T).T)).T + t
        z = pc[:, 2]
        view = image.copy()
        h, w = view.shape[:2]
        ok = (np.isfinite(proj).all(axis=1) & (z > 0.2)
              & (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h))
        zz = z[ok]
        cc = cv2.applyColorMap(np.clip((zz - zz.min()) / (zz.max() - zz.min() + 1e-9) * 255,
                                       0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        for (u, v), c in zip(proj[ok].astype(int), cc):
            cv2.circle(view, (u, v), 1, tuple(int(x) for x in c[0]), -1)
        text = [f"t(mm)=({t[0]*1000:+.1f},{t[1]*1000:+.1f},{t[2]*1000:+.1f})  d_t={d_t:.1f}mm",
                f"d_R(deg)=({d_r:.2f})  in-view={int(ok.sum())}",
                "q/a tx w/s ty e/d tz | r/f rx t/g ry y/h rz",
                "z/x d_t  c/v d_r  p print  S save  ESC quit"]
        for i, line in enumerate(text):
            cv2.putText(view, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord("q"):
            t[0] += d_t / 1000.0
        elif key == ord("a"):
            t[0] -= d_t / 1000.0
        elif key == ord("w"):
            t[1] += d_t / 1000.0
        elif key == ord("s"):
            t[1] -= d_t / 1000.0
        elif key == ord("e"):
            t[2] += d_t / 1000.0
        elif key == ord("d"):
            t[2] -= d_t / 1000.0
        elif key == ord("r"):
            R = R @ rot_axis("x", d_r)
        elif key == ord("f"):
            R = R @ rot_axis("x", -d_r)
        elif key == ord("t"):
            R = R @ rot_axis("y", d_r)
        elif key == ord("g"):
            R = R @ rot_axis("y", -d_r)
        elif key == ord("y"):
            R = R @ rot_axis("z", d_r)
        elif key == ord("h"):
            R = R @ rot_axis("z", -d_r)
        elif key == ord("z"):
            d_t = max(0.5, d_t / 2)
        elif key == ord("x"):
            d_t = min(200.0, d_t * 2)
        elif key == ord("c"):
            d_r = max(0.01, d_r / 2)
        elif key == ord("v"):
            d_r = min(2.0, d_r * 2)
        elif key == ord("p"):
            print(f"t = {t.tolist()}")
            print(f"R = {R.flatten().tolist()}")
        elif key == ord("S"):
            shutil.copy(args.extrinsics, args.extrinsics + ".bak")
            data = {"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                        "translation": t.tolist()}}
            with open(args.extrinsics, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"saved {args.extrinsics} (backup {args.extrinsics}.bak)")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
