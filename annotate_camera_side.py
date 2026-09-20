#!/usr/bin/env python3
"""相机侧全量标注+坏角点统计: 对每帧跑与标定完全相同的角点检测/剔除策略。

每帧输出 <pose_dir>/camera_side_check.png:
  绿圈  = 参与 PnP 的角点(重投影误差<2.5px)
  红圈  = 被剔除的坏角点(标误差px), 整帧被剔除时画大红 X 并写 REJECTED
  蓝叉  = 当前位姿反投影的保留角点(应压在绿圈上)
  黄线  = 板外框四边形(内角点阵外扩1格纸边), 红十字 = 板中心投影
统计: 每帧状态/剔除详情/最终RMS; 末尾汇总哪些帧被整帧剔除、哪些帧内角点位置最常坏。

用法: python3 annotate_camera_side.py <帧目录...>
"""

import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_camera_lidar import (BOARD_COLS, board_object_points,
                                    find_board_corners, load_camera_info,
                                    solve_pnp_robust)

GREEN, BLUE, YELLOW, RED = (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)


def main() -> int:
    dirs = sys.argv[1:]
    if not dirs:
        print("用法: python3 annotate_camera_side.py <帧目录...>")
        return 1
    K, dist = load_camera_info("config/camera_info.yaml")
    obj = board_object_points()
    bad_counter: Counter = Counter()
    print(f"{'帧':<16} {'状态':<10} {'剔除角点(编号/误差)':<44} {'保留RMS'}")
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        img = cv2.imread(os.path.join(d, "color.png"))
        if img is None:
            print(f"{name:<16} {'无图像':<10}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            corners_all = cv2.cornerSubPix(
                gray, find_board_corners(img), (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            ).reshape(-1, 2)
        except ValueError:
            print(f"{name:<16} {'角点未检出':<10}")
            continue
        rvec, tvec, kept, rejected, truncated = solve_pnp_robust(obj, corners_all, K, dist)
        reproj = cv2.projectPoints(obj[kept], rvec, tvec, K, dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(((reproj - corners_all[kept]) ** 2).sum(axis=1).mean()))

        R, _ = cv2.Rodrigues(rvec)
        paper = np.array([[-0.12, -0.12, 0], [0.48, -0.12, 0], [0.48, 0.72, 0], [-0.12, 0.72, 0]])
        polygon = cv2.projectPoints(paper, rvec, tvec, K, dist)[0].reshape(-1, 2)
        center_cam = R @ np.array([0.18, 0.30, 0.0]) + tvec.ravel()
        cpx = cv2.projectPoints(center_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(2)

        vis = img.copy()
        cv2.polylines(vis, [polygon.astype(np.int32)], True, YELLOW, 2)
        for p in corners_all[kept].astype(int):
            cv2.circle(vis, tuple(p), 4, GREEN, 1)
        for p in reproj.astype(int):
            cv2.drawMarker(vis, tuple(p), BLUE, cv2.MARKER_TILTED_CROSS, 8, 1)
        for idx, e in rejected:
            p = corners_all[idx].astype(int)
            cv2.circle(vis, tuple(p), 8, RED, 2)
            cv2.putText(vis, f"{e:.1f}", (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
            bad_counter[f"#{idx}(r{idx // BOARD_COLS},c{idx % BOARD_COLS})"] += 1
        cu, cv = cpx.astype(int)
        cv2.drawMarker(vis, (cu, cv), RED, cv2.MARKER_CROSS, 16, 2)
        if truncated:
            cv2.line(vis, (30, 30), (70, 70), RED, 3)
            cv2.line(vis, (70, 30), (30, 70), RED, 3)
            cv2.putText(vis, "REJECTED >2 bad corners", (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
        out = os.path.join(d, "camera_side_check.png")
        cv2.imwrite(out, vis)

        status = "整帧剔除" if truncated else ("正常" if not rejected else f"剔{len(rejected)}个")
        rej = " ".join(f"#{i}:{e:.1f}px" for i, e in rejected)
        if truncated:
            rej += " (>2, 余者未列)"
        print(f"{name:<16} {status:<10} {rej:<44} {rms:.2f}px")

    if bad_counter:
        print("\n坏角点频率(帧内编号(行,列) → 出现帧数):")
        for k, v in bad_counter.most_common():
            print(f"  {k:<14} ×{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
