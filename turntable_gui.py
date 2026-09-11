#!/usr/bin/env python3
"""Turntable control GUI tool.

Usage:
    source install/setup.bash
    python3 turntable_gui.py
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2

SENSOR_ENV_LAUNCH = "ros2 launch perception_tower_sensor sensor_env.launch.py"
ORBBEC_USB_VENDOR_ID = "2bc5"
USBDEVFS_RESET_IOCTL = 0x5514

from perception_tower_sensor_interfaces.srv import TurntableCommand
from perception_tower_sensor_interfaces.msg import TurntableStatus

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext
    import tkinter.font as tkfont
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Error: tkinter is not installed")
    sys.exit(1)


def get_font(root=None):
    candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Arial"]
    available = set(tkfont.families(root=root))
    for name in candidates:
        if name in available:
            return name
    return "Arial"


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


@dataclass(frozen=True, slots=True)
class ScanCapture:
    raw_frames: list[tuple[float, PointCloud2]]
    angle_samples: list[tuple[float, float]]
    out_dir: str
    scan_start_deg: float
    scan_end_deg: float


class TurntableGuiController(Node):
    def __init__(self, color_topic: str, depth_topic: str, fairy_topic: str, config: dict):
        super().__init__("turntable_gui_controller")
        self._cfg = config
        self._executor: MultiThreadedExecutor | None = None
        self._no_fairy = config.get("no_fairy", False)
        self._rotation_axis = np.array(config.get("rotation_axis", [1.0, 0.0, 0.0]), dtype=np.float64)

        self.tt_cli = self.create_client(TurntableCommand, "/turntable/command")
        tt_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.tt_status_sub = self.create_subscription(
            TurntableStatus, "/turntable/status", self._on_tt_status, tt_qos
        )

        cam_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color_sub = self.create_subscription(Image, color_topic, self._on_color, cam_qos)
        self.depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, cam_qos)

        if not self._no_fairy:
            fairy_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
            self.fairy_sub = self.create_subscription(
                PointCloud2, fairy_topic, self._on_fairy, fairy_qos
            )
        else:
            self.fairy_sub = None

        self.tt_status: TurntableStatus | None = None
        self._capturing = False
        self._angle_samples: list[tuple[float, float]] = []
        self._raw_frames: list[tuple[float, PointCloud2]] = []
        self._fairy_msg_count = 0
        self._merge_thread: threading.Thread | None = None

        self.color_msg: Image | None = None
        self.color_ts: float | None = None
        self.depth_msg: Image | None = None
        self.depth_ts: float | None = None

        self._launch_proc: subprocess.Popen | None = None

        self._out_base = os.path.join(os.getcwd(), "turntable_output")
        self.out_dir = self._out_base
        os.makedirs(self._out_base, exist_ok=True)

        self._log_callback = None

    def _descendant_pids(self, root_pid: int) -> list[int]:
        children: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    stat = f.read()
            except OSError:
                continue
            paren = stat.rfind(")")
            if paren < 0:
                continue
            fields = stat[paren + 2:].split()
            try:
                ppid = int(fields[1])
            except (IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))

        descendants: list[int] = []
        stack = [root_pid]
        while stack:
            for child in children.get(stack.pop(), []):
                descendants.append(child)
                stack.append(child)
        return descendants

    def _signal_pids(self, pids: list[int], sig: int):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _signal_process_group(self, pgid: int | None, sig: int):
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _signal_tree(self, root_pid: int, pgid: int | None, sig: int):
        self._signal_pids(self._descendant_pids(root_pid), sig)
        self._signal_process_group(pgid, sig)
        try:
            os.kill(root_pid, sig)
        except ProcessLookupError:
            pass

    def _wait_process(self, proc: subprocess.Popen, timeout_s: float) -> bool:
        try:
            proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _orbbec_usb_device_paths(self) -> list[str]:
        paths: list[str] = []
        base = "/sys/bus/usb/devices"
        try:
            entries = os.listdir(base)
        except OSError:
            return paths
        for entry in entries:
            dirpath = os.path.join(base, entry)
            try:
                with open(os.path.join(dirpath, "idVendor")) as f:
                    if f.read().strip() != ORBBEC_USB_VENDOR_ID:
                        continue
                with open(os.path.join(dirpath, "busnum")) as f:
                    busnum = int(f.read().strip())
                with open(os.path.join(dirpath, "devnum")) as f:
                    devnum = int(f.read().strip())
            except (OSError, ValueError):
                continue
            paths.append(f"/dev/bus/usb/{busnum:03d}/{devnum:03d}")
        return paths

    def reset_usb_cameras(self) -> int:
        reset_count = 0
        for path in self._orbbec_usb_device_paths():
            try:
                fd = os.open(path, os.O_WRONLY)
            except OSError as exc:
                self.log(f"USB reset: cannot open {path}: {exc}")
                continue
            try:
                fcntl.ioctl(fd, USBDEVFS_RESET_IOCTL, 0)
                reset_count += 1
                self.log(f"USB reset ok: {path}")
            except OSError as exc:
                self.log(f"USB reset failed: {path}: {exc}")
            finally:
                os.close(fd)
        if reset_count == 0:
            self.log("USB reset: no Orbbec camera found")
        return reset_count

    def _terminate_launch_process(self):
        proc = self._launch_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        self._signal_tree(proc.pid, pgid, signal.SIGINT)
        if self._wait_process(proc, 8.0):
            return
        self._signal_tree(proc.pid, pgid, signal.SIGTERM)
        if self._wait_process(proc, 5.0):
            return
        self._signal_tree(proc.pid, pgid, signal.SIGKILL)
        self._wait_process(proc, 2.0)

    def cleanup(self):
        self._log_callback = None
        self.log("Cleaning up background services...")
        if self._merge_thread is not None and self._merge_thread.is_alive():
            self.log("Waiting for background merge to finish...")
            self._merge_thread.join(timeout=30.0)
            if self._merge_thread.is_alive():
                self.log("Background merge did not finish in time")
        if self._launch_proc is not None and self._launch_proc.poll() is None:
            self._terminate_launch_process()
            self.log("Background services stopped")

    def set_log_callback(self, callback):
        self._log_callback = callback

    def log(self, msg: str):
        self.get_logger().info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except tk.TclError:
                self._log_callback = None

    def _on_tt_status(self, msg: TurntableStatus):
        self.tt_status = msg
        local_time = time.monotonic()
        if self._capturing:
            self._angle_samples.append((local_time, msg.angle_deg))

    def _on_color(self, msg: Image):
        self.color_msg = msg
        self.color_ts = time.monotonic()

    def _on_depth(self, msg: Image):
        self.depth_msg = msg
        self.depth_ts = time.monotonic()

    def _on_fairy(self, msg: PointCloud2):
        self._fairy_msg_count += 1
        if not self._capturing:
            return
        local_time = time.monotonic()
        self._raw_frames.append((local_time, msg))

    def _wait_for_service_ready(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.tt_cli.service_is_ready():
                return True
            self._executor.spin_once(timeout_sec=0.01)
        return False

    def call_turntable(self, cmd: int, target_deg: float = 0.0,
                       duration_s: float = 0.0, timeout_s: float = 15.0) -> bool:
        if not self._wait_for_service_ready(timeout_s=5.0):
            self.log("Service /turntable/command is not available")
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
            self.log(f"[service] Timeout (cmd={cmd})")
            return False
        result = fut.result()
        self.log(f"[service] Response: success={result.success} msg={result.message}")
        return result.success

    def wait_for_turntable_idle(self, target_deg: float | None = None,
                                tol_deg: float = 0.2, timeout_s: float = 60.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            self._executor.spin_once(timeout_sec=0.01)
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
            self.log(f"[wait] In position: angle={angle:.2f}deg elapsed={time.monotonic()-start:.1f}s")
            return True
        self.log("[wait] Timeout!")
        return False

    def _wait_for_state(self, state: int, timeout_s: float = 10.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            self._executor.spin_once(timeout_sec=0.01)
            if self.tt_status is not None and self.tt_status.state == state:
                return True
        return False

    def wait_for_photo(self, freshness_s: float = 0.5, max_gap_s: float = 0.2,
                       timeout_s: float = 30.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            self._executor.spin_once(timeout_sec=0.05)
            now = time.monotonic()
            if self.color_msg is None or self.depth_msg is None:
                continue
            c_age = now - self.color_ts
            d_age = now - self.depth_ts
            if c_age <= freshness_s and d_age <= freshness_s:
                if abs(self.color_ts - self.depth_ts) <= max_gap_s:
                    return self.color_msg, self.depth_msg
            time.sleep(0.02)
        self.log("[photo] Timeout")
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
            self.log(f"Saved color: {cpath}")
            self.log(f"Saved depth: {dpath}")
        except Exception as exc:
            self.log(f"cv_bridge/cv2 unavailable ({exc}), saving raw bytes")
            with open(cpath + ".raw", "wb") as f:
                f.write(bytes(color.data))
            with open(dpath + ".raw", "wb") as f:
                f.write(bytes(depth.data))

    def reset_to_ready(self, ready_deg: float = 90.0, sweep_speed: float = 40.0):
        self.log("=== Reset to 90 deg ===")
        self.log("1) Homing...")
        if not self.call_turntable(TurntableCommand.Request.CMD_HOME, timeout_s=30.0):
            self.log("Home failed")
            return False
        if not self.wait_for_turntable_idle(timeout_s=60.0):
            self.log("Home timeout")
            return False
        self.log("Home complete")

        self.log(f"2) Move to {ready_deg}deg...")
        delta = abs(ready_deg)
        duration = max(0.5, delta / sweep_speed)
        if not self.call_turntable(TurntableCommand.Request.CMD_MOVE, ready_deg, duration):
            self.log("Move failed")
            return False
        if not self.wait_for_turntable_idle(target_deg=ready_deg, timeout_s=60.0):
            self.log("Move timeout")
            return False
        self.log(f"Reset to {ready_deg}deg")
        return True

    def scan(self, scan_range_deg: float, ready_deg: float = 90.0, sweep_speed: float = 40.0):
        if self._merge_thread is not None and self._merge_thread.is_alive():
            self.log("Waiting for previous merge to finish before starting new scan...")
            self._merge_thread.join()
            self.log("Previous merge finished")

        self.out_dir = os.path.join(
            self._out_base, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "frames"), exist_ok=True)
        self.log(f"Output directory for this scan: {self.out_dir}")

        if self.tt_status is not None and abs(self.tt_status.angle_deg - ready_deg) > 0.5:
            self.log(
                f"Not at ready position (current={self.tt_status.angle_deg:.2f}deg), "
                f"moving to {ready_deg}deg before scan..."
            )
            delta = abs(self.tt_status.angle_deg - ready_deg)
            duration = max(0.5, delta / sweep_speed)
            if not self.call_turntable(TurntableCommand.Request.CMD_MOVE, ready_deg, duration):
                self.log("Move to ready position failed")
                return False
            if not self.wait_for_turntable_idle(target_deg=ready_deg, timeout_s=60.0):
                self.log("Move to ready position timeout")
                return False
            self.log("At ready position")

        scan_start_deg = ready_deg - scan_range_deg
        scan_end_deg = ready_deg + scan_range_deg
        self.log(f"=== Scan {scan_start_deg}deg ~ {scan_end_deg}deg ===")

        self.log("1) Take pre-scan photo at ready position...")
        photo = self.wait_for_photo()
        if photo is None:
            self.log("Pre-scan photo failed")
            return False
        self.save_photos(*photo)
        self.log("Pre-scan photo saved")

        self.log(f"2) Move to scan start {scan_start_deg}deg...")
        delta = abs(scan_start_deg - ready_deg)
        duration = max(0.5, delta / sweep_speed)
        if not self.call_turntable(TurntableCommand.Request.CMD_MOVE, scan_start_deg, duration):
            self.log("Move to scan start failed")
            return False
        if not self.wait_for_turntable_idle(target_deg=scan_start_deg, timeout_s=60.0):
            self.log("Move to scan start timeout")
            return False
        self.log("At scan start")

        self.log("3) Start scan move...")
        scan_duration = max(0.5, abs(scan_end_deg - scan_start_deg) / sweep_speed)
        if not self.call_turntable(TurntableCommand.Request.CMD_MOVE, scan_end_deg, scan_duration):
            self.log("Scan move failed")
            return False

        self.log("Waiting for motion to start...")
        if not self._wait_for_state(TurntableStatus.STATE_MOVING, timeout_s=10.0):
            self.log("Motion did not start")
            return False

        self.log("Motion started, begin capture")
        self._start_capture()

        self.log("Waiting for motion to finish...")
        if not self.wait_for_turntable_idle(target_deg=scan_end_deg, timeout_s=60.0):
            self.log("Scan move timeout")
            self._stop_capture()
            return False

        self._stop_capture()
        self.log("Scan complete")

        self.log("4) Processing data and returning to ready in parallel...")
        capture = ScanCapture(
            raw_frames=list(self._raw_frames),
            angle_samples=list(self._angle_samples),
            out_dir=self.out_dir,
            scan_start_deg=scan_start_deg,
            scan_end_deg=scan_end_deg,
        )
        merge_thread = threading.Thread(
            target=self._process_frames_and_merge,
            args=(capture,),
            daemon=True,
        )
        self._merge_thread = merge_thread
        merge_thread.start()

        delta = abs(scan_end_deg - ready_deg)
        duration = max(0.5, delta / sweep_speed)
        return_success = self.call_turntable(TurntableCommand.Request.CMD_MOVE, ready_deg, duration)
        if return_success:
            return_success = self.wait_for_turntable_idle(target_deg=ready_deg, timeout_s=60.0)

        if not return_success:
            self.log("Return to ready failed")
            return False
        self.log("Returned to ready")
        return True

    def _start_capture(self):
        self._angle_samples.clear()
        self._raw_frames.clear()
        self._capturing = True
        self.log("Capture started")

    def _stop_capture(self):
        self._capturing = False
        self.log("Capture stopped")

    def _process_frames_and_merge(self, capture: ScanCapture):
        captured_frames: list[tuple[float, np.ndarray]] = []
        if not self._no_fairy and capture.raw_frames:
            self.log(f"Processing {len(capture.raw_frames)} frames...")
            for stamp, msg in capture.raw_frames:
                points = self._pointcloud2_to_numpy(msg)
                if points is not None and len(points) > 0:
                    captured_frames.append((stamp, points))
                    ply_path = os.path.join(
                        capture.out_dir, "frames", f"frame_{len(captured_frames):04d}.ply"
                    )
                    self._save_ply(ply_path, points)
            self.log(f"Processed {len(captured_frames)} frames")
        self.stitch_and_save_merged(captured_frames, capture)

    def stitch_and_save_merged(
        self,
        captured_frames: list[tuple[float, np.ndarray]],
        capture: ScanCapture,
    ):
        if not captured_frames or not capture.angle_samples:
            self.log("No data to merge")
            return

        axis = self._rotation_axis / np.linalg.norm(self._rotation_axis)
        angle_times = np.array([s[0] for s in capture.angle_samples])
        angle_values = np.array([s[1] for s in capture.angle_samples])

        all_points = []
        for frame_stamp, frame_xyz in captured_frames:
            angle_deg = np.interp(frame_stamp, angle_times, angle_values)
            if capture.scan_start_deg <= angle_deg <= capture.scan_end_deg:
                transformed = transform_frame(frame_xyz, angle_deg, axis)
                all_points.append(transformed)

        if not all_points:
            self.log("No frames in scan range to merge")
            return

        merged = np.concatenate(all_points, axis=0)
        out_path = os.path.join(capture.out_dir, "merged.ply")
        self._save_ply(out_path, merged)
        self.log(f"Merged point cloud saved: {out_path}")
        self.log(f"Merged frames: {len(all_points)}, total points: {len(merged)}")

        photo_angle = (capture.scan_start_deg + capture.scan_end_deg) / 2.0
        self._colorize_merged(capture.out_dir, photo_angle)

    def _colorize_merged(self, out_dir: str, photo_angle_deg: float):
        extrinsics = os.path.join(os.getcwd(), "config", "camera_extrinsics.yaml")
        camera_info = os.path.join(os.getcwd(), "config", "camera_info.yaml")
        if not (os.path.exists(extrinsics) and os.path.exists(camera_info)):
            self.log("Colorize skipped: config/camera_extrinsics.yaml or camera_info.yaml missing")
            return
        try:
            from colorize_pointcloud import colorize
            out, total, colored = colorize(out_dir, extrinsics, camera_info, photo_angle_deg)
            self.log(f"Colored point cloud saved: {out} ({colored}/{total} points)")
        except Exception as exc:
            self.log(f"Colorize failed: {exc}")

    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray | None:
        try:
            import struct
            fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
            if not all(name in fields for name in ('x', 'y', 'z')):
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
        except Exception:
            return None

    def _save_ply(self, path: str, xyz: np.ndarray):
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

    def check_ready(self, color_topic: str, depth_topic: str, fairy_topic: str) -> tuple[bool, str]:
        checks = []
        ready = True

        if self.tt_cli.service_is_ready():
            checks.append("Turntable service: ready")
        else:
            checks.append("Turntable service: not ready")
            ready = False

        if self.tt_status is not None:
            checks.append(f"Turntable status: connected (angle={self.tt_status.angle_deg:.2f}deg)")
        else:
            checks.append("Turntable status: not connected")
            ready = False

        if self.color_msg is not None and (time.monotonic() - self.color_ts) <= 2.0:
            checks.append("Color image: has data")
        else:
            checks.append("Color image: no data")
            ready = False

        if self.depth_msg is not None and (time.monotonic() - self.depth_ts) <= 2.0:
            checks.append("Depth image: has data")
        else:
            checks.append("Depth image: no data")
            ready = False

        if self._no_fairy:
            checks.append("LiDAR: skipped")
        else:
            if self._fairy_msg_count > 0:
                checks.append(f"LiDAR: has data ({self._fairy_msg_count} frames)")
            else:
                checks.append("LiDAR: no data")
                ready = False

        return ready, "; ".join(checks)

    def launch_sensor_env(self, timeout_s: float = 120.0) -> bool:
        proc = self._launch_proc
        if proc is None or proc.poll() is not None:
            self.reset_usb_cameras()
            time.sleep(2.0)
            self.log(f"Starting background services: {SENSOR_ENV_LAUNCH}")
            try:
                proc = subprocess.Popen(
                    SENSOR_ENV_LAUNCH.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self._launch_proc = proc
            except Exception as e:
                self.log(f"Failed to start: {e}")
                return False

            def _reader():
                for line in proc.stdout:
                    self.log(f"[launch] {line.rstrip()}")
            threading.Thread(target=_reader, daemon=True).start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.log("Background launch process exited")
                return False
            self._executor.spin_once(timeout_sec=0.1)
            if self.tt_cli.service_is_ready() and self.tt_status is not None:
                self.log("Background services ready")
                return True
            time.sleep(0.5)
        self.log("Timeout waiting for background services")
        return False


class TurntableGuiApp:
    def __init__(self, node: TurntableGuiController, executor: MultiThreadedExecutor,
                 ready_deg: float, sweep_speed: float,
                 color_topic: str, depth_topic: str, fairy_topic: str):
        self.node = node
        self.executor = executor
        self.ready_deg = ready_deg
        self.sweep_speed = sweep_speed
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.fairy_topic = fairy_topic

        self.root = tk.Tk()
        self.root.title("Perception Tower Turntable Control")
        self.root.geometry("700x520")

        self.font_name = get_font(self.root)
        self.font_large = (self.font_name, 14)
        self.font_normal = (self.font_name, 11)

        self._create_widgets()
        node.set_log_callback(self._log)
        self._set_buttons(tk.DISABLED, tk.DISABLED)
        self._update_status(False, "Checking background services...")
        self._run_in_thread(self._startup_readiness_loop)

    def _create_widgets(self):
        title = tk.Label(self.root, text="Perception Tower Turntable Control", font=(self.font_name, 20))
        title.pack(pady=15)

        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5)
        self.status_canvas = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=5)
        self.status_dot = self.status_canvas.create_oval(2, 2, 14, 14, fill="red")
        self.status_label = tk.Label(status_frame, text="Not Ready", font=self.font_normal)
        self.status_label.pack(side=tk.LEFT)

        range_frame = tk.Frame(self.root)
        range_frame.pack(pady=10)

        tk.Label(range_frame, text="Scan range (each side, max 90):", font=self.font_normal).pack(side=tk.LEFT)
        self.range_var = tk.StringVar(value="60")
        self.range_entry = tk.Entry(range_frame, textvariable=self.range_var, width=8, font=self.font_normal)
        self.range_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(range_frame, text="deg", font=self.font_normal).pack(side=tk.LEFT)

        self.reset_btn = tk.Button(
            self.root, text="1. Reset to 90deg", font=self.font_large,
            width=25, height=2, command=self._on_reset, state=tk.DISABLED
        )
        self.reset_btn.pack(pady=10)

        self.scan_btn = tk.Button(
            self.root, text="2. Start Scan", font=self.font_large,
            width=25, height=2, command=self._on_scan, state=tk.DISABLED
        )
        self.scan_btn.pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=80, height=14, font=("Consolas", 10)
        )
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def _update_status(self, ready: bool, text: str):
        color = "green" if ready else "red"
        self.status_canvas.itemconfig(self.status_dot, fill=color)
        self.status_label.config(text=text)

    def _startup_readiness_loop(self):
        while True:
            ready, detail = self.node.check_ready(self.color_topic, self.depth_topic, self.fairy_topic)
            self.root.after(0, lambda d=detail: self._log(d))
            if ready:
                self.root.after(0, lambda: self._update_status(True, "READY"))
                self.root.after(0, lambda: self._set_buttons(tk.NORMAL, tk.DISABLED))
                self.root.after(0, lambda: self._log("Background services ready"))
                return
            self.root.after(0, lambda: self._update_status(False, "Not Ready - Starting background services..."))
            if not self.node.launch_sensor_env(timeout_s=120.0):
                self.root.after(0, lambda: self._update_status(False, "Background services failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to start background services. Please check the environment and restart."))
                return
            time.sleep(1.0)

    def _log(self, msg: str):
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)

    def _get_scan_range(self) -> float | None:
        try:
            value = float(self.range_var.get())
            if value <= 0 or value > 90:
                raise ValueError("must be between 1 and 90")
            return value
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid scan range: {e}")
            return None

    def _run_in_thread(self, func):
        def wrapper():
            try:
                func()
            except Exception as e:
                self.root.after(0, lambda: self._log(f"Error: {e}"))
        threading.Thread(target=wrapper, daemon=True).start()

    def _set_buttons(self, reset_state, scan_state):
        self.reset_btn.config(state=reset_state)
        self.scan_btn.config(state=scan_state)

    def _on_reset(self):
        self._set_buttons(tk.DISABLED, tk.DISABLED)
        self._run_in_thread(self._do_reset)

    def _do_reset(self):
        success = self.node.reset_to_ready(self.ready_deg, self.sweep_speed)
        self.root.after(0, lambda: self._after_reset(success))

    def _after_reset(self, success: bool):
        if success:
            self._set_buttons(tk.NORMAL, tk.NORMAL)
            self._log("Reset complete. Set scan range and click Start Scan.")
        else:
            self._set_buttons(tk.NORMAL, tk.DISABLED)
            messagebox.showerror("Error", "Reset failed")

    def _on_scan(self):
        scan_range = self._get_scan_range()
        if scan_range is None:
            return
        self._set_buttons(tk.DISABLED, tk.DISABLED)
        self._run_in_thread(lambda: self._do_scan(scan_range))

    def _do_scan(self, scan_range: float):
        success = self.node.scan(scan_range, self.ready_deg, self.sweep_speed)
        self.root.after(0, lambda: self._after_scan(success))

    def _after_scan(self, success: bool):
        self._set_buttons(tk.NORMAL, tk.NORMAL)
        if success:
            self._log(f"Output directory: {self.node.out_dir}")
            messagebox.showinfo("Done", f"Scan complete\nOutput directory: {self.node.out_dir}")
        else:
            messagebox.showerror("Error", "Scan failed")

    def run(self):
        def _on_sigint(signum, frame):
            self.root.after(0, self.root.destroy)

        def _poll_signals():
            self.root.after(100, _poll_signals)

        signal.signal(signal.SIGINT, _on_sigint)
        self.root.after(100, _poll_signals)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Perception Tower Turntable Control GUI")
    parser.add_argument("--color-topic", default="/camera/color/image_raw", help="Color image topic")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw", help="Depth image topic")
    parser.add_argument("--fairy-topic", default="/rslidar_points", help="LiDAR point cloud topic")
    parser.add_argument("--ready-deg", type=float, default=90.0, help="Ready angle")
    parser.add_argument("--sweep-speed", type=float, default=40.0, help="Sweep speed")
    parser.add_argument("--no-fairy", action="store_true", help="Skip LiDAR capture")
    parser.add_argument("--rotation-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Merge rotation axis")
    args = parser.parse_args()

    config = {
        "no_fairy": args.no_fairy,
        "rotation_axis": list(args.rotation_axis),
    }

    rclpy.init()
    node = TurntableGuiController(args.color_topic, args.depth_topic, args.fairy_topic, config)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    node._executor = executor

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        app = TurntableGuiApp(
            node, executor, args.ready_deg, args.sweep_speed,
            args.color_topic, args.depth_topic, args.fairy_topic
        )
        app.run()
    finally:
        try:
            node.cleanup()
        except Exception as exc:
            node.get_logger().error(f"cleanup failed: {exc}")
        executor.shutdown()
        executor_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
