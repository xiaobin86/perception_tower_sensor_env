#!/usr/bin/env python3
"""find_board_corners 检测阶梯单测。

测两件事:
1. (RED->GREEN) 三张低对比度/倾斜失败帧 (070700/070957/083911) 能检出 8x6=48 角点;
2. (回归) 已通过的帧检出结果与基线完全一致(角点坐标逐位相同), 增强阶梯不得改变已有帧的路径。

基线生成: python3 test_find_board_corners.py --save-baseline
(用当前生产代码记录所有可检出帧的角点 -> /tmp 基线文件)
"""

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_camera_lidar import (BOARD_COLS, BOARD_ROWS, board_object_points,
                                    find_board_corners, load_camera_info)

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        ".corner_baseline.npz")
FAILING = ["20260923_070700", "20260923_070957", "20260923_083911"]
# (6,8) pattern 命中的帧: 角点必须转置成 8/行 与矩形物体点一致, 否则 PnP 崩溃
TRANSPOSED = ["20260923_083911"]


def _detect(path: str) -> np.ndarray:
    img = cv2.imread(path)
    assert img is not None, f"读不到图像: {path}"
    corners = find_board_corners(img)
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    assert c.shape == (BOARD_COLS * BOARD_ROWS, 2), f"{path}: 角点数 {c.shape}"
    assert np.isfinite(c).all(), f"{path}: 角点含非法值"
    return c


def save_baseline() -> None:
    base = {}
    for d in sorted(glob.glob("turntable_output/20260923_*")):
        p = os.path.join(d, "color.png")
        if not os.path.exists(p):
            continue
        name = os.path.basename(d)
        try:
            base[name] = _detect(p)
            print(f"  基线 {name}: OK")
        except (ValueError, AssertionError) as exc:
            print(f"  基线 {name}: 跳过 ({exc})")
    np.savez(BASELINE, **base)
    print(f"基线已存 {BASELINE} ({len(base)} 帧)")


def test_failing_frames_detected() -> None:
    """三张曾失败的帧现在必须能检出 48 个角点。"""
    for name in FAILING:
        d = os.path.join("turntable_output", name, "color.png")
        if not os.path.exists(d):
            print(f"  {name}: 目录不存在, 跳过")
            continue
        _detect(d)
        print(f"  {name}: 检出 OK")


def test_passing_frames_unchanged() -> None:
    """已检出帧的角点坐标必须与基线逐位一致(同路径返回, 增强阶梯不得影响)。"""
    assert os.path.exists(BASELINE), "先运行 --save-baseline 生成基线"
    base = np.load(BASELINE)
    n_checked = 0
    for name in base.files:
        d = os.path.join("turntable_output", name, "color.png")
        if not os.path.exists(d):
            continue
        c = _detect(d)
        ref = base[name]
        # 坐标必须完全一致(同一路径命中), 至少也要亚像素级一致
        err = np.abs(c - ref).max()
        assert err < 1e-6, f"{name}: 角点被改变, max|Δ|={err}"
        n_checked += 1
    print(f"  回归 {n_checked} 帧一致")


def test_transposed_pattern_ordered() -> None:
    """(6,8) 命中帧: 返回角点必须与 8/行 物体点对应(PnP RMS < 15px, 错配时 ~50px)。"""
    K, dist = load_camera_info("config/camera_info.yaml")
    obj = board_object_points()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for name in TRANSPOSED:
        d = os.path.join("turntable_output", name, "color.png")
        if not os.path.exists(d):
            print(f"  {name}: 目录不存在, 跳过")
            continue
        img = cv2.imread(d)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(
            gray, find_board_corners(img), (5, 5), (-1, -1), criteria).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        assert ok, f"{name}: solvePnP 失败"
        reproj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(((reproj - corners) ** 2).sum(axis=1).mean()))
        assert rms < 15.0, f"{name}: PnP RMS {rms:.1f}px — (6,8) 角点顺序未转置?"
        print(f"  {name}: 顺序正确 (PnP RMS {rms:.2f}px)")


def test_detection_deterministic() -> None:
    """同一帧重复检出必须逐位一致(cv2 棋盘格检测走全局 RNG, 未固定种子时
    ~1/10 概率角点跳 ~14px — 073136 实测)。"""
    for name in ["20260923_073136", "20260923_083911"]:
        d = os.path.join("turntable_output", name, "color.png")
        if not os.path.exists(d):
            continue
        first = _detect(d)
        for _ in range(4):
            err = np.abs(_detect(d) - first).max()
            assert err < 1e-6, f"{name}: 重复检出不一致, max|Δ|={err}"
        print(f"  {name}: 5 次检出一致")


if __name__ == "__main__":
    if "--save-baseline" in sys.argv:
        save_baseline()
        sys.exit(0)
    test_failing_frames_detected()
    test_transposed_pattern_ordered()
    test_detection_deterministic()
    test_passing_frames_unchanged()
    print("ALL PASS")
