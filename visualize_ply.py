#!/usr/bin/env python3
"""LiDAR点云拼合可视化工具。

用法：
    python3 visualize_ply.py
    python3 visualize_ply.py /mnt/d/work/perception-tower/turntable_output/20260907_181431
    python3 visualize_ply.py /mnt/d/work/perception-tower/turntable_output/20260907_181431 --axis 0 1 0
    python3 visualize_ply.py /mnt/d/work/perception-tower/turntable_output/20260907_181431 --manual
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import open3d as o3d

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


def select_folder_gui() -> str | None:
    """使用GUI选择文件夹。"""
    if not HAS_TKINTER:
        return None
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder = filedialog.askdirectory(
        title="选择LiDAR数据目录",
        initialdir="/mnt/d/work/perception-tower/turntable_output"
    )
    root.destroy()
    
    return folder if folder else None


def load_alignment_result(csv_path: str) -> list[dict]:
    """加载帧与角度的映射文件。"""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'frame': int(row['frame']),
                'stamp_sec': float(row['stamp_sec']),
                'angle_deg': float(row['angle_deg']),
                'num_points': int(row['num_points']),
            })
    return results


def load_ply(ply_path: str) -> np.ndarray | None:
    """加载PLY文件为numpy数组。"""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        return np.asarray(pcd.points)
    except Exception as e:
        print(f"加载PLY失败: {e}")
        return None


def rotation_matrix_x(deg: float) -> np.ndarray:
    """绕X轴旋转。"""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(deg: float) -> np.ndarray:
    """绕Y轴旋转。"""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕任意轴旋转的旋转矩阵。"""
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.radians(-angle_deg)
    ux, uy, uz = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)   ],
    ])
    return R


def get_world_coord_frame_lines(size: float = 0.5) -> o3d.geometry.LineSet:
    """获取线框世界坐标系。"""
    points = [
        [0, 0, 0],
        [size, 0, 0],
        [0, size, 0],
        [0, 0, size]
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_lidar_coord_frame(size: float = 0.3) -> o3d.geometry.TriangleMesh:
    """获取经过帧内变换后的雷达坐标系。"""
    R_y = rotation_matrix_y(-90.0)
    R_x = rotation_matrix_x(0.0)
    R_total = R_y @ R_x
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    coord.rotate(R_total, center=[0, 0, 0])
    return coord


def transform_frame(points: np.ndarray, angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """先按角度绕轴旋转，再做帧内变换（Y轴顺时针90度，X轴0度）。"""
    R_y = rotation_matrix_y(-90.0)
    R_x = rotation_matrix_x(0.0)
    R_axis = rotation_matrix_from_axis_angle(axis, angle_deg)
    return points @ R_axis.T @ R_y.T @ R_x.T


def stitch_frames(frames: list[tuple[float, np.ndarray]], 
                  angles: list[float], 
                  axis: np.ndarray) -> np.ndarray:
    """将多帧点云按旋转角度拼合。"""
    all_points = []
    
    for i, (frame, angle) in enumerate(zip(frames, angles)):
        rotated = transform_frame(frame, angle, axis)
        all_points.append(rotated)
    
    if all_points:
        return np.concatenate(all_points, axis=0)
    return np.array([])


def visualize_open3d(merged: np.ndarray, title: str = "LiDAR点云拼合"):
    """使用Open3D可视化。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    
    colors = np.zeros((len(merged), 3))
    z_min, z_max = merged[:, 2].min(), merged[:, 2].max()
    if z_max > z_min:
        colors[:, 0] = (merged[:, 2] - z_min) / (z_max - z_min)
        colors[:, 1] = 0.5
        colors[:, 2] = 1.0 - (merged[:, 2] - z_min) / (z_max - z_min)
    else:
        colors[:, 0] = 0.5
        colors[:, 1] = 0.5
        colors[:, 2] = 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"点云数量: {len(merged)}")
    print(f"坐标范围: X=[{merged[:, 0].min():.2f}, {merged[:, 0].max():.2f}]")
    print(f"           Y=[{merged[:, 1].min():.2f}, {merged[:, 1].max():.2f}]")
    print(f"           Z=[{merged[:, 2].min():.2f}, {merged[:, 2].max():.2f}]")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd)
    
    coord_frame_world = get_world_coord_frame_lines(size=0.5)
    coord_frame_lidar = get_lidar_coord_frame(size=0.3)
    vis.add_geometry(coord_frame_world)
    vis.add_geometry(coord_frame_lidar)
    
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_view_control().set_up([0, 0, 1])
    vis.get_view_control().set_front([0, -1, 0])
    vis.run()
    vis.destroy_window()


def visualize_manual_mode(data_dir: str, alignment: list[dict], axis: np.ndarray):
    print("\n=== 手动模式 ===")
    print("按 N: 叠加下一帧")
    print("按 R: 重置")
    print("按 Q: 退出")
    print("================\n")

    merged_points = []
    frame_idx = [0]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR点云拼合", width=1200, height=800)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_view_control().set_up([0, 0, 1])
    vis.get_view_control().set_front([0, -1, 0])

    coord_frame_world = get_world_coord_frame_lines(size=0.5)
    coord_frame_lidar = get_lidar_coord_frame(size=0.3)
    vis.add_geometry(coord_frame_world)
    vis.add_geometry(coord_frame_lidar)

    empty = np.zeros((0, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(empty)
    pcd.colors = o3d.utility.Vector3dVector(empty)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    def update_display():
        if merged_points:
            merged = np.concatenate(merged_points, axis=0)
        else:
            merged = np.zeros((0, 3))

        pcd.points = o3d.utility.Vector3dVector(merged)

        if len(merged) > 0:
            colors = np.zeros((len(merged), 3))
            z_min, z_max = merged[:, 2].min(), merged[:, 2].max()
            if z_max > z_min:
                colors[:, 0] = (merged[:, 2] - z_min) / (z_max - z_min)
                colors[:, 1] = 0.5
                colors[:, 2] = 1.0 - (merged[:, 2] - z_min) / (z_max - z_min)
            else:
                colors[:, 0] = 0.5
                colors[:, 1] = 0.5
                colors[:, 2] = 0.5
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        vis.update_geometry(pcd)
        vis.reset_view_point(True)
        vis.get_view_control().set_up([0, 0, 1])
        vis.get_view_control().set_front([0, -1, 0])
        vis.poll_events()
        vis.update_renderer()

    def next_frame(vis):
        if frame_idx[0] >= len(alignment):
            print("所有帧已拼合完成")
            return False

        entry = alignment[frame_idx[0]]
        frame_num = entry['frame']
        angle_deg = entry['angle_deg']

        ply_path = os.path.join(data_dir, "frames", f"frame_{frame_num:04d}.ply")
        if os.path.exists(ply_path):
            points = load_ply(ply_path)
            if points is not None:
                rotated = transform_frame(points, angle_deg, axis)
                merged_points.append(rotated)

                update_display()

                merged = np.concatenate(merged_points, axis=0)
                print(f"帧 {frame_idx[0]+1}/{len(alignment)}: angle={angle_deg:.2f}°, total={len(merged)}")

        frame_idx[0] += 1
        return True

    def reset_frame(vis):
        merged_points.clear()
        frame_idx[0] = 0
        update_display()
        print("已重置")
        return True

    def quit_callback(vis):
        vis.close()
        return False

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("R"), reset_frame)
    vis.register_key_callback(ord("Q"), quit_callback)

    print("按 N 开始叠加第一帧...")
    vis.run()
    vis.destroy_window()


def visualize_auto_mode(data_dir: str, alignment: list[dict], axis: np.ndarray):
    """全自动模式：一次拼合所有帧。"""
    print("\n=== 全自动模式 ===")
    print(f"共 {len(alignment)} 帧")
    print("================\n")
    
    frames = []
    angles = []
    
    for entry in alignment:
        frame_num = entry['frame']
        angle_deg = entry['angle_deg']
        
        ply_path = os.path.join(data_dir, "frames", f"frame_{frame_num:04d}.ply")
        if not os.path.exists(ply_path):
            print(f"帧 {frame_num} 文件不存在: {ply_path}")
            continue
        
        points = load_ply(ply_path)
        if points is None:
            continue
        
        frames.append(points)
        angles.append(angle_deg)
        print(f"加载帧 {frame_num}: angle={angle_deg:.2f}°, points={len(points)}")
    
    if not frames:
        print("没有可用的帧")
        return
    
    print(f"\n拼合 {len(frames)} 帧...")
    merged = stitch_frames(frames, angles, axis)
    print(f"拼合完成，总点数: {len(merged)}")
    
    visualize_open3d(merged, f"LiDAR拼合结果 - 旋转轴=[{axis[0]:.1f},{axis[1]:.1f},{axis[2]:.1f}]")


def main():
    parser = argparse.ArgumentParser(description="LiDAR点云拼合可视化")
    parser.add_argument("data_dir", nargs='?', help="数据目录（不指定则弹出选择对话框）")
    parser.add_argument("--axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="旋转轴 (x y z)，默认X轴")
    parser.add_argument("--manual", action="store_true", help="手动模式，按N拼合一帧")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if data_dir is None:
        if HAS_TKINTER:
            print("未指定目录，弹出文件夹选择对话框...")
            data_dir = select_folder_gui()
        else:
            print("错误: 未安装tkinter，请直接指定目录路径")
            sys.exit(1)
    
    if not data_dir:
        print("未选择目录，退出")
        sys.exit(0)
    
    # 检查数据目录
    if not os.path.isdir(data_dir):
        print(f"错误: 目录不存在: {data_dir}")
        sys.exit(1)
    
    csv_path = os.path.join(data_dir, "alignment_result.csv")
    if not os.path.exists(csv_path):
        print(f"错误: 映射文件不存在: {csv_path}")
        sys.exit(1)
    
    frames_dir = os.path.join(data_dir, "frames")
    if not os.path.isdir(frames_dir):
        print(f"错误: frames目录不存在: {frames_dir}")
        sys.exit(1)
    
    # 加载映射文件
    alignment = load_alignment_result(csv_path)
    print(f"加载 {len(alignment)} 帧映射")
    
    # 旋转轴
    axis = np.array(args.axis, dtype=np.float64)
    print(f"旋转轴: [{axis[0]:.1f}, {axis[1]:.1f}, {axis[2]:.1f}]")
    
    # 执行模式
    if args.manual:
        visualize_manual_mode(data_dir, alignment, axis)
    else:
        visualize_auto_mode(data_dir, alignment, axis)


if __name__ == "__main__":
    main()
