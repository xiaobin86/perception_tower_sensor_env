#!/usr/bin/env python3
"""colorize_pointcloud 取色采样单元测试(合成数据, 无需真实帧)。

验证:
  1. sample_colors nearest 模式与原 np.round 行为一致;
  2. sample_colors bilinear 模式在已知梯度图上给出解析正确的插值;
  3. bilinear 越界钳制不崩溃, 掩码外点保持默认灰色;
  4. colorize() 端到端: 合成 PLY + 合成图像, bilinear 取到正确颜色。

运行: python3 test_colorize_pointcloud.py
"""

from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from colorize_pointcloud import colorize, sample_colors, write_ply_rgb


def gradient_image(h: int = 100, w: int = 100) -> np.ndarray:
    """R=2u, G=2v, B=128 的确定性梯度图(uint8, BGR)。双线性插值可解析验证。"""
    u = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = (2 * u).astype(np.uint8)   # R = 2u
    img[:, :, 1] = (2 * v).astype(np.uint8)   # G = 2v
    img[:, :, 0] = 128                        # B = 128
    return img


class TestSampleColors(unittest.TestCase):
    def setUp(self) -> None:
        self.img = gradient_image()
        self.h, self.w = self.img.shape[:2]

    def test_nearest_matches_rounding(self) -> None:
        u = np.array([10.3, 20.6, 55.0])
        v = np.array([20.7, 30.2, 60.0])
        mask = np.array([True, True, True])
        colors = sample_colors(self.img, u, v, mask, "nearest")
        # 原实现: np.round 后索引 ( bankers rounding: 55.0->55, 20.6->21, 30.2->30 )
        expected = np.array([
            self.img[21, 10],  # round(20.7)=21, round(10.3)=10
            self.img[30, 21],  # round(30.2)=30, round(20.6)=21
            self.img[60, 55],
        ], dtype=np.uint8)[:, ::-1]  # BGR -> RGB
        np.testing.assert_array_equal(colors, expected)

    def test_bilinear_exact_on_gradient(self) -> None:
        # 梯度图 R=2u G=2v 上双线性插值解析可算: R=2u, G=2v (亚像素值保留)
        u = np.array([10.3])
        v = np.array([20.7])
        mask = np.array([True])
        colors = sample_colors(self.img, u, v, mask, "bilinear")
        self.assertEqual(colors.shape, (1, 3))
        # uint8 输出: round(2*10.3)=round(20.6)=21, round(2*20.7)=round(41.4)=41, B=128
        np.testing.assert_array_equal(colors[0], np.array([21, 41, 128], dtype=np.uint8))

    def test_bilinear_differs_from_nearest(self) -> None:
        u = np.array([10.3])
        v = np.array([20.7])
        mask = np.array([True])
        near = sample_colors(self.img, u, v, mask, "nearest")
        bil = sample_colors(self.img, u, v, mask, "bilinear")
        self.assertFalse(np.array_equal(near, bil))  # nearest: (20,40,128), bilinear: (21,41,128)

    def test_bilinear_border_clamp(self) -> None:
        # 右下角外侧的点: 钳制到最后一行/列, 不得崩溃或回绕
        u = np.array([self.w - 0.4, self.w + 5.0])
        v = np.array([self.h - 0.4, self.h + 5.0])
        mask = np.array([True, True])
        colors = sample_colors(self.img, u, v, mask, "bilinear")
        np.testing.assert_array_equal(colors[0], np.array([2 * (self.w - 1), 2 * (self.h - 1), 128]))
        np.testing.assert_array_equal(colors[1], colors[0])

    def test_mask_out_stays_gray(self) -> None:
        u = np.array([10.0, 50.0])
        v = np.array([10.0, 50.0])
        mask = np.array([True, False])
        colors = sample_colors(self.img, u, v, mask, "bilinear")
        np.testing.assert_array_equal(colors[1], np.array([30, 30, 30], dtype=np.uint8))

    def test_invalid_sampling_raises(self) -> None:
        with self.assertRaises(ValueError):
            sample_colors(self.img, np.array([1.0]), np.array([1.0]), np.array([True]), "cubic")


class TestColorizeEndToEnd(unittest.TestCase):
    def test_colorize_bilinear_on_synthetic_pose(self) -> None:
        """合成 pose 目录: 简单内参/外参下投一个已知点, bilinear 取色正确。"""
        with tempfile.TemporaryDirectory() as pose:
            img = gradient_image()
            cv2.imwrite(os.path.join(pose, "color.png"), img)
            # Z=2 平面上的点: K=diag(100,100), c=(50,50), R=I, t=0 → u=100*X/2+50
            points = np.array([
                [2 * (10.3 - 50) / 100, 2 * (20.7 - 50) / 100, 2.0],  # → (10.3, 20.7)
                [2 * (70.5 - 50) / 100, 2 * (80.25 - 50) / 100, 2.0],  # → (70.5, 80.25)
            ])
            write_ply_rgb(os.path.join(pose, "merged.ply"), points,
                          np.full((len(points), 3), 30, dtype=np.uint8))

            camera_info = {"k": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0], "d": [0.0] * 5}
            extrinsics = {"lidar_to_camera": {
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0]}}
            import yaml
            with open(os.path.join(pose, "cam.yaml"), "w") as f:
                yaml.safe_dump(camera_info, f)
            with open(os.path.join(pose, "ext.yaml"), "w") as f:
                yaml.safe_dump(extrinsics, f)

            out, total, colored = colorize(pose, os.path.join(pose, "ext.yaml"),
                                           os.path.join(pose, "cam.yaml"), photo_angle=0.0,
                                           sampling="bilinear", output="out.ply")
            self.assertEqual((total, colored), (2, 2))
            with open(out) as f:
                for header_lines, line in enumerate(f, start=1):
                    if line.strip() == "end_header":
                        break
            got = np.loadtxt(out, skiprows=header_lines)[:, 3:6]
            # round-half-even: 2*80.25=160.5 → 160
            np.testing.assert_array_equal(got, np.array([[21, 41, 128], [141, 160, 128]]))


if __name__ == "__main__":
    unittest.main()
