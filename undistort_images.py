#!/usr/bin/env python3
"""根据相机内参对照片进行去畸变处理。

用法：
    python3 undistort_images.py /workspace/turntable_output/20260907_181431

输入：
    config/camera_info.yaml
    color.png
    depth.png

输出：
    color_undistorted.png
    depth_undistorted.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def undistort_image(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                    new_K: np.ndarray | None = None) -> np.ndarray:
    """对图像去畸变。"""
    h, w = image.shape[:2]
    if new_K is None:
        new_K = K.copy()
    
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="图像去畸变")
    parser.add_argument("data_dir", help="数据目录")
    args = parser.parse_args()
    
    camera_info_path = os.path.join(args.data_dir, "config", "camera_info.yaml")
    color_path = os.path.join(args.data_dir, "color.png")
    depth_path = os.path.join(args.data_dir, "depth.png")
    
    if not os.path.exists(camera_info_path):
        print(f"错误: {camera_info_path} 不存在")
        print("请先运行 save_camera_info.py 保存相机内参")
        sys.exit(1)
    
    if not os.path.exists(color_path):
        print(f"错误: {color_path} 不存在")
        sys.exit(1)
    
    camera_info = load_yaml(camera_info_path)
    K = np.array(camera_info["k"]).reshape(3, 3)
    D = np.array(camera_info["d"])
    
    print(f"加载相机内参: {camera_info['width']}x{camera_info['height']}")
    print(f"K:\n{K}")
    print(f"D: {D}")
    
    # 彩色图去畸变
    color = cv2.imread(color_path)
    color_undistorted = undistort_image(color, K, D)
    color_out = os.path.join(args.data_dir, "color_undistorted.png")
    cv2.imwrite(color_out, color_undistorted)
    print(f"已保存: {color_out}")
    
    # 深度图去畸变
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_undistorted = undistort_image(depth, K, D)
        depth_out = os.path.join(args.data_dir, "depth_undistorted.png")
        cv2.imwrite(depth_out, depth_undistorted)
        print(f"已保存: {depth_out}")
    else:
        print(f"警告: {depth_path} 不存在，跳过深度图去畸变")


if __name__ == "__main__":
    main()
