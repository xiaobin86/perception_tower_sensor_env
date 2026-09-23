#!/usr/bin/env python3
"""dense_colorize_pointcloud LiDAR→RGBD D2C 管线的单元测试(合成数据)。

验证:
  1. splat_depth: 同像素多深度取最近(min-z); dilate 溅射 2x2;
  2. fill_holes_nearest: 空洞由最近有效像素填补, 超 max_fill_px 保持无效;
  3. unproject: 常值深度平面反投影回射线几何正确;
  4. 端到端: 稀疏栅格点投到梯度图, 稠密输出覆盖且颜色解析正确。

运行: python3 test_dense_colorize_pointcloud.py
"""

from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from colorize_pointcloud import write_ply_rgb
from dense_colorize_pointcloud import (dense_colorize, fill_holes_nearest,
                                       splat_depth, unproject)


def gradient_image(h: int = 60, w: int = 80) -> np.ndarray:
    u = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = (2 * u).astype(np.uint8)
    img[:, :, 1] = (2 * v).astype(np.uint8)
    img[:, :, 0] = 128
    return img


class TestSplatDepth(unittest.TestCase):
    def test_min_z_wins_conflict(self) -> None:
        depth = splat_depth(np.array([10.0, 10.4]), np.array([10.0, 10.4]),
                            np.array([5.0, 2.0]), (20, 20), dilate=False)
        self.assertEqual(depth[10, 10], 2.0)  # 最近点胜出

    def test_dilate_fills_2x2(self) -> None:
        depth = splat_depth(np.array([10.0]), np.array([10.0]), np.array([3.0]),
                            (20, 20), dilate=True)
        block = depth[10:12, 10:12]
        np.testing.assert_array_equal(block, np.full((2, 2), 3.0))
        self.assertFalse(np.isfinite(depth[12, 10]))  # 2x2 之外不受影响


class TestFillHoles(unittest.TestCase):
    def test_nearest_fill_and_radius_limit(self) -> None:
        depth = np.full((10, 10), np.inf, dtype=np.float32)
        depth[0, 0] = 1.0
        depth[9, 9] = 2.0
        filled, valid = fill_holes_nearest(depth, max_fill_px=5.0)
        # (0,1) 离 (0,0) 1px → 被填 1.0; (5,5) 离两有效点都 >5px(对角约6.4/5.7) → 保持无效
        self.assertEqual(filled[0, 1], 1.0)
        self.assertTrue(valid[0, 1])
        self.assertFalse(valid[5, 5])
        # 有效像素原值不动
        self.assertEqual(filled[9, 9], 2.0)


class TestUnproject(unittest.TestCase):
    def test_constant_depth_plane(self) -> None:
        h, w = 4, 6
        K = np.array([[100.0, 0.0, (w - 1) / 2], [0.0, 100.0, (h - 1) / 2], [0.0, 0.0, 1.0]])
        depth = np.full((h, w), 2.0, dtype=np.float32)
        valid = np.ones((h, w), dtype=bool)
        pts = unproject(depth, valid, K, np.zeros(5))
        self.assertEqual(pts.shape, (h * w, 3))
        # 像素(2,1): ((2-2.5),(1-1.5))*2/100 = (-0.01,-0.01,2); 像素(0,0): (-0.05,-0.03,2)
        center = pts[1 * w + 2]
        corner = pts[0]
        np.testing.assert_allclose(center, [-0.01, -0.01, 2.0], atol=1e-6)
        np.testing.assert_allclose(corner, [-0.05, -0.03, 2.0], atol=1e-6)


class TestDenseColorizeEndToEnd(unittest.TestCase):
    def test_dense_pipeline_on_synthetic_pose(self) -> None:
        h, w = 60, 80
        with tempfile.TemporaryDirectory() as pose:
            cv2.imwrite(os.path.join(pose, "color.png"), gradient_image(h, w))
            # Z=2 平面上的稀疏栅格 LiDAR 点(间隔4px), K=100,c=中心, R=I, t=0
            K = np.array([[100.0, 0.0, (w - 1) / 2], [0.0, 100.0, (h - 1) / 2], [0.0, 0.0, 1.0]])
            us, vs = np.meshgrid(np.arange(2, w - 2, 4.0), np.arange(2, h - 2, 4.0))
            us, vs = us.ravel(), vs.ravel()
            xs = (us - K[0, 2]) * 2.0 / K[0, 0]
            ys = (vs - K[1, 2]) * 2.0 / K[1, 1]
            points = np.column_stack([xs, ys, np.full(len(us), 2.0)])
            write_ply_rgb(os.path.join(pose, "merged.ply"), points,
                          np.full((len(points), 3), 30, dtype=np.uint8))
            import yaml
            with open(os.path.join(pose, "cam.yaml"), "w") as f:
                yaml.safe_dump({"k": K.ravel().tolist(), "d": [0.0] * 5}, f)
            with open(os.path.join(pose, "ext.yaml"), "w") as f:
                yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": np.eye(3).tolist(),
                                                    "translation": [0.0, 0.0, 0.0]}}, f)

            out, n_pts, coverage, _ = dense_colorize(
                pose, os.path.join(pose, "ext.yaml"), os.path.join(pose, "cam.yaml"),
                photo_angle=0.0, dilate=True, max_fill_px=3.0, output="dense.ply")
            self.assertGreater(coverage, 0.85)  # 稀疏点经膨胀+填补后覆盖率应很高
            with open(out) as f:
                for header_lines, line in enumerate(f, start=1):
                    if line.strip() == "end_header":
                        break
            data = np.loadtxt(out, skiprows=header_lines)
            pts3d, colors = data[:, :3], data[:, 3:6]
            # 所有点应在 Z=2 平面上(最近邻填补误差 ≤ 2px 对应的深度偏差)
            np.testing.assert_allclose(pts3d[:, 2], 2.0, atol=0.1)
            # 颜色解析: RGB=(2u,2v,128), u=100*x/z+30 → 恢复像素坐标核对
            u_back = 100.0 * pts3d[:, 0] / pts3d[:, 2] + (w - 1) / 2
            v_back = 100.0 * pts3d[:, 1] / pts3d[:, 2] + (h - 1) / 2
            np.testing.assert_allclose(colors[:, 0], np.round(2 * u_back), atol=1.0)
            np.testing.assert_allclose(colors[:, 1], np.round(2 * v_back), atol=1.0)
            np.testing.assert_array_equal(colors[:, 2], 128)


class TestIndexMap(unittest.TestCase):
    def test_index_map_points_to_correct_rows(self) -> None:
        h, w = 60, 80
        with tempfile.TemporaryDirectory() as pose:
            cv2.imwrite(os.path.join(pose, "color.png"), gradient_image(h, w))
            K = np.array([[100.0, 0.0, (w - 1) / 2], [0.0, 100.0, (h - 1) / 2], [0.0, 0.0, 1.0]])
            us, vs = np.meshgrid(np.arange(2, w - 2, 4.0), np.arange(2, h - 2, 4.0))
            xs = (us.ravel() - K[0, 2]) * 2.0 / K[0, 0]
            ys = (vs.ravel() - K[1, 2]) * 2.0 / K[1, 1]
            points = np.column_stack([xs, ys, np.full(len(us.ravel()), 2.0)])
            write_ply_rgb(os.path.join(pose, "merged.ply"), points,
                          np.full((len(points), 3), 30, dtype=np.uint8))
            import yaml
            with open(os.path.join(pose, "cam.yaml"), "w") as f:
                yaml.safe_dump({"k": K.ravel().tolist(), "d": [0.0] * 5}, f)
            with open(os.path.join(pose, "ext.yaml"), "w") as f:
                yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": np.eye(3).tolist(),
                                                    "translation": [0.0, 0.0, 0.0]}}, f)

            out, n, _, _ = dense_colorize(
                pose, os.path.join(pose, "ext.yaml"), os.path.join(pose, "cam.yaml"),
                photo_angle=0.0, output="dense.ply", save_index_map=True)
            index_map = np.load(os.path.join(pose, "dense_index.npy"))
            self.assertEqual(index_map.shape, (h, w))
            self.assertEqual(index_map.dtype, np.int32)

            with open(out) as f:
                for header_lines, line in enumerate(f, start=1):
                    if line.strip() == "end_header":
                        break
            data = np.loadtxt(out, skiprows=header_lines)
            colors = data[:, 3:6]
            valid = index_map >= 0
            self.assertEqual(valid.sum(), n)
            # index_map 指向的 PLY 行必须恰好是按光栅序排列的: 第 k 个有效像素 → 第 k 行
            rows = index_map[valid]
            np.testing.assert_array_equal(rows, np.arange(n))
            # 抽查若干像素: 该像素颜色 == PLY 对应行颜色
            img = gradient_image(h, w)
            for v, u in [(10, 2), (30, 40), (50, 77)]:
                if not valid[v, u]:
                    continue
                np.testing.assert_array_equal(colors[index_map[v, u]], img[v, u][::-1])
            # 无效位为 -1 且出现在无效像素上
            self.assertTrue((index_map[~valid] == -1).all())


class TestBijectionCheck(unittest.TestCase):
    def test_output_is_one_point_per_pixel(self) -> None:
        h, w = 40, 50
        with tempfile.TemporaryDirectory() as pose:
            cv2.imwrite(os.path.join(pose, "color.png"), gradient_image(h, w))
            K = np.array([[100.0, 0.0, (w - 1) / 2], [0.0, 100.0, (h - 1) / 2], [0.0, 0.0, 1.0]])
            us, vs = np.meshgrid(np.arange(2, w - 2, 3.0), np.arange(2, h - 2, 3.0))
            xs = (us.ravel() - K[0, 2]) * 2.0 / K[0, 0]
            ys = (vs.ravel() - K[1, 2]) * 2.0 / K[1, 1]
            points = np.column_stack([xs, ys, np.full(len(us.ravel()), 2.0)])
            write_ply_rgb(os.path.join(pose, "merged.ply"), points,
                          np.full((len(points), 3), 30, dtype=np.uint8))
            import yaml
            with open(os.path.join(pose, "cam.yaml"), "w") as f:
                yaml.safe_dump({"k": K.ravel().tolist(), "d": [0.0] * 5}, f)
            with open(os.path.join(pose, "ext.yaml"), "w") as f:
                yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": np.eye(3).tolist(),
                                                    "translation": [0.0, 0.0, 0.0]}}, f)

            out, n, cov, check = dense_colorize(
                pose, os.path.join(pose, "ext.yaml"), os.path.join(pose, "cam.yaml"),
                photo_angle=0.0, output="dense.ply")
            self.assertTrue(check["ok"])
            self.assertEqual(check["unique_pixels"], n)
            self.assertLess(check["max_offset_px"], 0.05)


if __name__ == "__main__":
    unittest.main()
