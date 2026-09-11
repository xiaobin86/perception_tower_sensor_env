#!/usr/bin/env python3
"""控制转盘转动并采集LiDAR数据，保存为PLY文件。

用法：
    source install/setup.bash
    python3 turntable_control.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2

from perception_tower_sensor_interfaces.srv import TurntableCommand
from perception_tower_sensor_interfaces.msg import TurntableStatus


def rotation_matrix_x(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
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


def transform_frame(points: np.ndarray, angle_deg: float, axis: np.ndarray) -> np.ndarray:
    R_y = rotation_matrix_y(-90.0)
    R_x = rotation_matrix_x(0.0)
    R_axis = rotation_matrix_from_axis_angle(axis, angle_deg)
    return points @ R_axis.T @ R_y.T @ R_x.T


class TurntableController(Node):
    def __init__(self, color_topic: str, depth_topic: str, fairy_topic: str, config: dict):
        super().__init__("turntable_controller")
        self._cfg = config
        self._executor: MultiThreadedExecutor | None = None
        self._no_fairy = config.get("no_fairy", False)

        # 转盘服务
        self.tt_cli = self.create_client(TurntableCommand, "/turntable/command")
        tt_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.tt_status_sub = self.create_subscription(
            TurntableStatus, "/turntable/status", self._on_tt_status, tt_qos
        )

        cam_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color_sub = self.create_subscription(Image, color_topic, self._on_color, cam_qos)
        self.depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, cam_qos)

        if not self._no_fairy:
            fairy_qos = QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.fairy_sub = self.create_subscription(
                PointCloud2, fairy_topic, self._on_fairy, fairy_qos
            )
        else:
            self.fairy_sub = None

        # 状态变量
        self.tt_status: TurntableStatus | None = None
        self._fairy_frame_count = 0
        self._capturing = False

        self.color_msg: Image | None = None
        self.color_ts: float | None = None
        self.depth_msg: Image | None = None
        self.depth_ts: float | None = None

        # 角度记录 (使用header时间戳)
        self._angle_samples: list[tuple[float, float]] = []  # (stamp_sec, angle_deg)

        # 采集的帧 (使用header时间戳)
        self._raw_frames: list[tuple[float, PointCloud2]] = []  # (stamp_sec, raw_msg)
        self._captured_frames: list[tuple[float, np.ndarray]] = []  # (stamp_sec, xyz)

        # 输出目录
        self.out_dir = os.path.join(
            os.getcwd(), "turntable_output", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "frames"), exist_ok=True)
        self.get_logger().info(f"输出目录: {self.out_dir}")

    def _on_tt_status(self, msg: TurntableStatus):
        self.tt_status = msg
        
        local_time = time.monotonic()
        
        if self._capturing:
            self._angle_samples.append((local_time, msg.angle_deg))
            if len(self._angle_samples) % 10 == 0:
                self.get_logger().info(
                    f"[角度] #{len(self._angle_samples)} angle={msg.angle_deg:.2f}° local={local_time:.3f}"
                )
        elif not hasattr(self, '_tt_status_log_done'):
            self.get_logger().info(
                f"[转盘状态] state={msg.state} angle={msg.angle_deg:.2f}° done={msg.done}"
            )
            self._tt_status_log_done = True

    def _on_color(self, msg: Image):
        self.color_msg = msg
        self.color_ts = time.monotonic()
        if not hasattr(self, '_color_log_done'):
            self.get_logger().info(f"[color] 收到: {msg.width}x{msg.height} {msg.encoding}")
            self._color_log_done = True

    def _on_depth(self, msg: Image):
        self.depth_msg = msg
        self.depth_ts = time.monotonic()
        if not hasattr(self, '_depth_log_done'):
            self.get_logger().info(f"[depth] 收到: {msg.width}x{msg.height} {msg.encoding}")
            self._depth_log_done = True

    def _on_fairy(self, msg: PointCloud2):
        if not self._capturing:
            return
        
        local_time = time.monotonic()
        self._raw_frames.append((local_time, msg))
        
        if len(self._raw_frames) % 10 == 0:
            self.get_logger().info(
                f"[LiDAR] 帧 #{len(self._raw_frames)} local={local_time:.3f}"
            )

    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray | None:
        """将PointCloud2转换为numpy数组(xyz)。"""
        try:
            import struct
            
            fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
            
            if not all(name in fields for name in ('x', 'y', 'z')):
                self.get_logger().error("点云缺少x/y/z字段")
                return None
            
            point_step = msg.point_step
            points = []
            data = msg.data
            num_points = len(data) // point_step
            
            for i in range(num_points):
                offset = i * point_step
                x = struct.unpack_from('f', data, offset + fields['x'][0])[0]
                y = struct.unpack_from('f', data, offset + fields['y'][0])[0]
                z = struct.unpack_from('f', data, offset + fields['z'][0])[0]
                
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points.append([x, y, z])
            
            if points:
                return np.array(points, dtype=np.float32)
            return None
        except Exception as e:
            self.get_logger().error(f"点云转换失败: {e}")
            return None

    def _save_ply(self, path: str, xyz: np.ndarray):
        """保存点云为PLY文件。"""
        n = xyz.shape[0]
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            np.savetxt(f, xyz, fmt="%.6f")

    def _wait_for_service_ready(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.tt_cli.service_is_ready():
                return True
            self._executor.spin_once(timeout_sec=0.01)
        return False

    def call_turntable(self, cmd: int, target_deg: float = 0.0,
                       duration_s: float = 0.0, timeout_s: float = 15.0) -> bool:
        """发送service call，等回复确认命令已接收。"""
        if not self._wait_for_service_ready(timeout_s=5.0):
            self.get_logger().error("service /turntable/command 不可用")
            return False
        req = TurntableCommand.Request()
        req.command = cmd
        req.target_deg = target_deg
        req.duration_s = duration_s
        fut = self.tt_cli.call_async(req)
        deadline = time.monotonic() + timeout_s
        while not fut.done() and time.monotonic() < deadline:
            self._executor.spin_once(timeout_sec=0.01)
        if not fut.done():
            self.get_logger().error(f"[service] 超时 (cmd={cmd})")
            return False
        result = fut.result()
        self.get_logger().info(f"[service] 回复: success={result.success} msg={result.message}")
        return result.success

    def wait_for_turntable_idle(self, target_deg: float | None = None,
                                tol_deg: float = 0.2, timeout_s: float = 60.0) -> bool:
        start = time.monotonic()
        last_log = start
        while time.monotonic() - start < timeout_s:
            self._executor.spin_once(timeout_sec=0.01)
            now = time.monotonic()
            if now - last_log >= 1.0:
                last_log = now
                if self.tt_status:
                    self.get_logger().info(
                        f"[等待] state={self.tt_status.state} angle={self.tt_status.angle_deg:.2f}° "
                        f"done={self.tt_status.done} target={target_deg}° elapsed={now-start:.1f}s"
                    )
                else:
                    self.get_logger().info(f"[等待] 无状态 elapsed={now-start:.1f}s")
            if self.tt_status is None:
                continue
            state = self.tt_status.state
            angle = self.tt_status.angle_deg
            done = getattr(self.tt_status, "done", True)
            if state != TurntableStatus.STATE_IDLE:
                continue
            if target_deg is not None and abs(angle - target_deg) > tol_deg:
                continue
            if not done:
                continue
            self.get_logger().info(f"[等待] 到位: angle={angle:.2f}° elapsed={time.monotonic()-start:.1f}s")
            return True
        self.get_logger().error(f"[等待] 超时! 最后 state={self.tt_status.state if self.tt_status else '?'} "
                                f"angle={self.tt_status.angle_deg if self.tt_status else '?'}°")
        return False

    def wait_for_photo(self, freshness_s: float = 0.5, max_gap_s: float = 0.2,
                       timeout_s: float = 30.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            self._executor.spin_once(timeout_sec=0.05)
            now = time.monotonic()
            if self.color_msg is None or self.depth_msg is None:
                if int(now - start) % 5 == 0 and int(now - start) != getattr(self, '_last_photo_log', -1):
                    self._last_photo_log = int(now - start)
                    self.get_logger().info(
                        f"[photo] 等待中... color={'有' if self.color_msg else '无'} "
                        f"depth={'有' if self.depth_msg else '无'}"
                    )
                continue
            c_age = now - self.color_ts
            d_age = now - self.depth_ts
            if c_age <= freshness_s and d_age <= freshness_s:
                if abs(self.color_ts - self.depth_ts) <= max_gap_s:
                    return self.color_msg, self.depth_msg
            time.sleep(0.02)
        self.get_logger().error(
            f"[photo] 超时: color={'有' if self.color_msg else '无'} "
            f"depth={'有' if self.depth_msg else '无'}"
        )
        return None

    def save_photos(self, color: Image, depth: Image):
        cpath = os.path.join(self.out_dir, "color.png")
        dpath = os.path.join(self.out_dir, "depth.png")
        try:
            from cv_bridge import CvBridge
            import cv2
            bridge = CvBridge()
            color_cv = bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
            depth_cv = bridge.imgmsg_to_cv2(depth, desired_encoding="16UC1")
            cv2.imwrite(cpath, color_cv)
            cv2.imwrite(dpath, depth_cv)
            self.get_logger().info(f"已保存 color: {cpath}")
            self.get_logger().info(f"已保存 depth: {dpath}")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge/cv2 不可用 ({exc})，保存 raw bytes")
            with open(cpath + ".raw", "wb") as f:
                f.write(bytes(color.data))
            with open(dpath + ".raw", "wb") as f:
                f.write(bytes(depth.data))

    def start_capture(self):
        """开始采集。"""
        self._fairy_frame_count = 0
        self._angle_samples.clear()
        self._raw_frames.clear()
        self._captured_frames.clear()
        self._capturing = True
        self.get_logger().info("开始采集")

    def stop_capture(self):
        """停止采集并处理所有帧。"""
        self._capturing = False
        
        if not self._no_fairy and self._raw_frames:
            self.get_logger().info(f"处理 {len(self._raw_frames)} 帧...")
            for i, (stamp, msg) in enumerate(self._raw_frames):
                points = self._pointcloud2_to_numpy(msg)
                if points is not None and len(points) > 0:
                    self._fairy_frame_count += 1
                    self._captured_frames.append((stamp, points))
                    
                    ply_path = os.path.join(self.out_dir, "frames", f"frame_{self._fairy_frame_count:04d}.ply")
                    self._save_ply(ply_path, points)
                    
                    if self._fairy_frame_count % 10 == 0:
                        self.get_logger().info(f"已处理 {self._fairy_frame_count} 帧")
            self.get_logger().info(f"处理完成，共 {self._fairy_frame_count} 帧")
        
        if self._angle_samples:
            angle_start = self._angle_samples[0][0]
            angle_end = self._angle_samples[-1][0]
            
            self.get_logger().info("=== 运动过程数据统计 ===")
            self.get_logger().info(f"角度样本数: {len(self._angle_samples)}")
            self.get_logger().info(f"角度样本时间范围: {angle_start:.3f} ~ {angle_end:.3f} ({angle_end-angle_start:.2f}s)")
            self.get_logger().info(f"角度样本率: {len(self._angle_samples)/(angle_end-angle_start):.1f} Hz")
            
            if self._captured_frames:
                frame_start = self._captured_frames[0][0]
                frame_end = self._captured_frames[-1][0]
                self.get_logger().info(f"LiDAR帧数: {len(self._captured_frames)}")
                self.get_logger().info(f"LiDAR帧时间范围: {frame_start:.3f} ~ {frame_end:.3f} ({frame_end-frame_start:.2f}s)")
                self.get_logger().info(f"LiDAR平均帧率: {len(self._captured_frames)/(frame_end-frame_start):.1f} Hz")

    def align_and_log(self, scan_start_deg: float, scan_end_deg: float):
        if not self._angle_samples:
            self.get_logger().warn("无角度样本")
            return
        
        first_t, first_a = self._angle_samples[0]
        last_t, last_a = self._angle_samples[-1]
        self.get_logger().info(f"角度范围: {first_a:.2f}° ~ {last_a:.2f}°")
        self.get_logger().info(f"角度时间范围: {first_t:.3f} ~ {last_t:.3f}")
        
        if self._no_fairy:
            angle_csv_path = os.path.join(self.out_dir, "angle_log.csv")
            with open(angle_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stamp_sec", "angle_deg"])
                for t, deg in self._angle_samples:
                    w.writerow([f"{t:.6f}", f"{deg:.6f}"])
            self.get_logger().info(f"已保存角度日志: {angle_csv_path}")
            return
        
        if not self._captured_frames:
            self.get_logger().warn("无采集帧")
            return
        
        self.get_logger().info(f"帧数={len(self._captured_frames)}, 角度样本数={len(self._angle_samples)}")
        
        angle_times = np.array([s[0] for s in self._angle_samples])
        angle_values = np.array([s[1] for s in self._angle_samples])
        
        results = []
        scan_min = min(scan_start_deg, scan_end_deg)
        scan_max = max(scan_start_deg, scan_end_deg)
        
        for frame_stamp, frame_xyz in self._captured_frames:
            angle_deg = np.interp(frame_stamp, angle_times, angle_values)
            
            if scan_min <= angle_deg <= scan_max:
                results.append({
                    'stamp': frame_stamp,
                    'angle_deg': angle_deg,
                    'num_points': len(frame_xyz),
                })
        
        results.sort(key=lambda x: x['stamp'])
        
        # 输出对齐结果
        self.get_logger().info(f"\n=== 时间对齐结果 (扫描阶段 {scan_start_deg}° ~ {scan_end_deg}°) ===")
        self.get_logger().info(f"{'帧号':>6} {'时间戳':>14} {'角度(°)':>10} {'点数':>8}")
        self.get_logger().info("-" * 42)
        for i, r in enumerate(results, 1):
            self.get_logger().info(
                f"{i:6d} {r['stamp']:14.3f} {r['angle_deg']:10.2f} {r['num_points']:8d}"
            )
        
        # 保存对齐结果到CSV
        csv_path = os.path.join(self.out_dir, "alignment_result.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "stamp_sec", "angle_deg", "num_points"])
            for i, r in enumerate(results, 1):
                w.writerow([i, f"{r['stamp']:.6f}", f"{r['angle_deg']:.6f}", r['num_points']])
        self.get_logger().info(f"已保存对齐结果: {csv_path}")

        # 保存角度日志
        angle_csv_path = os.path.join(self.out_dir, "angle_log.csv")
        with open(angle_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stamp_sec", "angle_deg"])
            for t, deg in self._angle_samples:
                w.writerow([f"{t:.6f}", f"{deg:.6f}"])
        self.get_logger().info(f"已保存角度日志: {angle_csv_path}")

    def stitch_and_save_merged(self, scan_start_deg: float, scan_end_deg: float,
                                rotation_axis: list[float] = [0.0, 1.0, 0.0]):
        """拼合所有帧并保存为单个PLY文件。"""
        if not self._captured_frames or not self._angle_samples:
            self.get_logger().warn("无数据可拼合")
            return
        
        axis = np.array(rotation_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        angle_times = np.array([s[0] for s in self._angle_samples])
        angle_values = np.array([s[1] for s in self._angle_samples])
        
        all_points = []
        scan_min = min(scan_start_deg, scan_end_deg)
        scan_max = max(scan_start_deg, scan_end_deg)
        
        for frame_stamp, frame_xyz in self._captured_frames:
            angle_deg = np.interp(frame_stamp, angle_times, angle_values)
            
            if scan_min <= angle_deg <= scan_max:
                transformed = transform_frame(frame_xyz, angle_deg, axis)
                all_points.append(transformed)
        
        if not all_points:
            self.get_logger().warn("扫描范围内无帧可拼合")
            return
        
        merged = np.concatenate(all_points, axis=0)
        
        out_path = os.path.join(self.out_dir, "merged.ply")
        self._save_ply(out_path, merged)
        
        self.get_logger().info(f"已保存拼合点云: {out_path}")
        self.get_logger().info(f"拼合帧数: {len(all_points)}, 总点数: {len(merged)}")
        self.get_logger().info(f"坐标范围: X=[{merged[:,0].min():.2f},{merged[:,0].max():.2f}] "
                               f"Y=[{merged[:,1].min():.2f},{merged[:,1].max():.2f}] "
                               f"Z=[{merged[:,2].min():.2f},{merged[:,2].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description="转盘控制与LiDAR采集")
    parser.add_argument("--color-topic", default="/camera/color/image_raw", help="彩色图像话题")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw", help="深度图像话题")
    parser.add_argument("--fairy-topic", default="/rslidar_points", help="LiDAR点云话题")
    parser.add_argument("--ready-deg", type=float, default=90.0, help="就绪角度")
    parser.add_argument("--scan-start-deg", type=float, default=30.0, help="扫描起始角度")
    parser.add_argument("--scan-end-deg", type=float, default=150.0, help="扫描结束角度")
    parser.add_argument("--sweep-speed", type=float, default=40.0, help="扫描速度(deg/s)")
    parser.add_argument("--no-home", action="store_true", help="跳过归位")
    parser.add_argument("--no-fairy", action="store_true", help="跳过LiDAR采集，只测转盘")
    parser.add_argument("--rotation-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="拼合时的旋转轴 (x y z)，默认X轴")
    args = parser.parse_args()

    config = {
        "ready_deg": args.ready_deg,
        "sweep_speed_deg_s": args.sweep_speed,
        "scan_start_deg": args.scan_start_deg,
        "scan_end_deg": args.scan_end_deg,
        "no_fairy": args.no_fairy,
    }

    rclpy.init()
    node = TurntableController(args.color_topic, args.depth_topic, args.fairy_topic, config)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    node._executor = executor

    try:
        # 步骤1: 转盘归位
        if not args.no_home:
            node.get_logger().info("=== 1) 归位 ===")
            if not node.call_turntable(TurntableCommand.Request.CMD_HOME, timeout_s=30.0):
                node.get_logger().error("归位命令失败")
                sys.exit(1)
            if not node.wait_for_turntable_idle(timeout_s=60.0):
                node.get_logger().error("归位超时")
                sys.exit(1)
            node.get_logger().info("归位完成")

        # 步骤2: 移动到就绪位置
        node.get_logger().info(f"=== 2) 移动到就绪位置 {args.ready_deg}° ===")
        delta = abs(args.ready_deg)
        duration = max(0.5, delta / args.sweep_speed)
        if not node.call_turntable(
            TurntableCommand.Request.CMD_MOVE, args.ready_deg, duration
        ):
            node.get_logger().error("移动命令失败")
            sys.exit(1)
        if not node.wait_for_turntable_idle(target_deg=args.ready_deg, timeout_s=60.0):
            node.get_logger().error("移动超时")
            sys.exit(1)
        node.get_logger().info("就绪位置到达")

        node.get_logger().info("=== 3) 静止2秒后拍照 ===")
        time.sleep(2.0)
        photo = node.wait_for_photo()
        if photo is None:
            node.get_logger().error("拍照失败")
            sys.exit(1)
        node.save_photos(*photo)
        node.get_logger().info("拍照完成")

        node.get_logger().info(f"=== 4) 移动到扫描起始 {args.scan_start_deg}° ===")
        delta = abs(args.scan_start_deg - args.ready_deg)
        duration = max(0.5, delta / args.sweep_speed)
        if not node.call_turntable(
            TurntableCommand.Request.CMD_MOVE, args.scan_start_deg, duration
        ):
            node.get_logger().error("移动到扫描起始失败")
            sys.exit(1)
        if not node.wait_for_turntable_idle(target_deg=args.scan_start_deg, timeout_s=60.0):
            node.get_logger().error("移动到扫描起始超时")
            sys.exit(1)
        node.get_logger().info("已到扫描起始位置")

        node.get_logger().info(f"=== 5) 扫描摆动 {args.scan_start_deg}° -> {args.scan_end_deg}° ===")

        scan_duration = max(0.5, abs(args.scan_end_deg - args.scan_start_deg) / args.sweep_speed)
        if not node.call_turntable(
            TurntableCommand.Request.CMD_MOVE, args.scan_end_deg, scan_duration
        ):
            node.get_logger().error("扫描摆动失败")
            sys.exit(1)

        node.start_capture()

        if not node.wait_for_turntable_idle(target_deg=args.scan_end_deg, timeout_s=60.0):
            node.get_logger().error("扫描摆动超时")
            sys.exit(1)

        time.sleep(1.0)

        node.get_logger().info("=== 6) 停止采集 ===")
        node.stop_capture()

        node.align_and_log(args.scan_start_deg, args.scan_end_deg)

        node.stitch_and_save_merged(args.scan_start_deg, args.scan_end_deg, list(args.rotation_axis))

        node.get_logger().info(f"=== 7) 回到复位位置 {args.ready_deg}° ===")
        delta = abs(args.scan_end_deg - args.ready_deg)
        duration = max(0.5, delta / args.sweep_speed)
        if not node.call_turntable(
            TurntableCommand.Request.CMD_MOVE, args.ready_deg, duration
        ):
            node.get_logger().error("复位移动失败")
            sys.exit(1)
        if not node.wait_for_turntable_idle(target_deg=args.ready_deg, timeout_s=60.0):
            node.get_logger().error("复位超时")
            sys.exit(1)
        node.get_logger().info("复位完成")

        node.get_logger().info("=== 完成 ===")
        node.get_logger().info(f"输出目录: {node.out_dir}")
        node.get_logger().info(f"LiDAR帧数: {node._fairy_frame_count}")

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
