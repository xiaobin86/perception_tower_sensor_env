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

from install_config import InstallConfig, rotation_matrix
from lakibeam_viewer import LakiBeamViewer, ScanPoint, scan_to_xy

SENSOR_ENV_LAUNCH = "ros2 launch perception_tower_sensor sensor_env.launch.py"
ORBBEC_USB_VENDOR_ID = "2bc5"
USBDEVFS_RESET_IOCTL = 0x5514
MIDDLE_12_RINGS = tuple(range(42, 43))
FAST_MOVE_DURATION_S = 0.2
MIN_RANGE_M = 0.20

from perception_tower_sensor_interfaces.srv import TurntableCommand
from perception_tower_sensor_interfaces.msg import TurntableStatus

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
    import tkinter.font as tkfont
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Error: tkinter is not installed")
    sys.exit(1)


def list_serial_ports() -> list[str]:
    """枚举本机 USB 串口，返回下拉标签 "设备路径 — 描述"。

    优先用 pyserial 的 list_ports（能拿到描述）；不可用时回退到
    glob /dev/ttyUSB* 与 /dev/ttyACM*。注意 devcontainer 的 /dev 是
    私有 tmpfs，插拔后重新枚举的设备不会出现，此时返回空列表。
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        import glob
        return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    labels: list[str] = []
    for info in sorted(list_ports.comports(), key=lambda i: i.device):
        if info.vid is None and not info.device.startswith(("/dev/ttyUSB", "/dev/ttyACM")):
            continue
        desc = (info.description or "").strip()
        labels.append(f"{info.device} — {desc}" if desc and desc != "n/a" else info.device)
    return labels


def port_from_label(label: str) -> str:
    """从下拉标签取回设备路径；允许手输，标签首段即路径。"""
    parts = label.strip().split()
    return parts[0] if parts else ""


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
        self._min_dist: float = float(config.get("min_dist", MIN_RANGE_M))
        self._max_dist: float = float(config.get("max_dist", 3.0))
        self._lidar_kind = config.get("lidar_kind", "fairy")
        self._lakibeam_port = int(config.get("lakibeam_port", 2368))
        self._install_config_path = config.get("install_config", "config/install_side_mount.yaml")
        self._turntable_port = str(config.get("turntable_port", "/dev/ttyUSB0"))
        self._lakibeam: LakiBeamViewer | None = None
        self._lb_frames: list[tuple[float, list[ScanPoint]]] = []
        self._lb_thread: threading.Thread | None = None
        self._lb_capturing = False

        self.tt_cli = self.create_client(TurntableCommand, "/turntable/command")
        tt_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.tt_status_sub = self.create_subscription(
            TurntableStatus, "/turntable/status", self._on_tt_status, tt_qos
        )

        cam_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color_sub = self.create_subscription(Image, color_topic, self._on_color, cam_qos)
        self.depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, cam_qos)
        self.depth_cloud_topic = config.get("depth_cloud_topic", "/camera/depth_registered/points")
        cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.depth_cloud_sub = self.create_subscription(
            PointCloud2, self.depth_cloud_topic, self._on_depth_cloud, cloud_qos
        )

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
        self.depth_cloud_msg: PointCloud2 | None = None
        self.depth_cloud_ts: float | None = None
        self.depth_cloud_snapshot: PointCloud2 | None = None
        self.depth_cloud_xyz: np.ndarray | None = None

        self._launch_proc: subprocess.Popen | None = None

        self._out_base = os.path.join(os.getcwd(), "turntable_output")
        self.out_dir = self._out_base
        os.makedirs(self._out_base, exist_ok=True)

        self._log_callback = None
        self._log_file = None

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

    def restart_sensor_env(self, use_fairy: bool) -> bool:
        self.log("Restarting background services...")
        self._terminate_launch_process()
        self._launch_proc = None
        return self.launch_sensor_env(use_fairy=use_fairy)

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
        self.close_lakibeam()

    def set_log_callback(self, callback):
        self._log_callback = callback

    def log(self, msg: str):
        self.get_logger().info(msg)
        try:
            if self._log_file is None:
                os.makedirs(self._out_base, exist_ok=True)
                self._log_file = open(os.path.join(self._out_base, "gui.log"), "a", encoding="utf-8")
            self._log_file.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}\n")
            self._log_file.flush()
        except Exception:
            self._log_file = None
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

    def _on_depth_cloud(self, msg: PointCloud2):
        self.depth_cloud_msg = msg
        self.depth_cloud_ts = time.monotonic()

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
        self.capture_depth_cloud_snapshot()

    def capture_depth_cloud_snapshot(self):
        script = os.path.join(os.getcwd(), "fetch_depth_cloud_once.py")
        self.log(f"Fetching depth cloud at 90deg in a separate process -> {self.out_dir}")
        try:
            proc = subprocess.run(
                [sys.executable, script, self.out_dir, "--topic", self.depth_cloud_topic],
                capture_output=True, text=True, timeout=30.0, cwd=os.getcwd(),
            )
            for line in (proc.stdout or "").strip().splitlines():
                self.log(f"depth cloud fetch: {line}")
            if proc.returncode != 0:
                err = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
                self.log(f"depth cloud fetch failed (rc={proc.returncode}): {err}")
        except Exception as exc:
            self.log(f"depth cloud subprocess error: {exc}")
        ts = self.depth_cloud_ts
        self.depth_cloud_snapshot = (self.depth_cloud_msg
                                     if ts is not None and (time.monotonic() - ts) <= 2.0 else None)

    def _fetch_depth_cloud(self, timeout_s: float = 8.0):
        node = rclpy.create_node("depth_cloud_fetch")
        received: dict = {}
        cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(PointCloud2, self.depth_cloud_topic,
                                 lambda m: received.setdefault("m", m), cloud_qos)
        deadline = time.monotonic() + timeout_s
        while "m" not in received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node()
        return received.get("m")

    def save_depth_cloud(self, fresh_timeout_s: float = 6.0, fresh_window_s: float = 2.0):
        if os.path.exists(os.path.join(self.out_dir, "depth_cloud.ply")):
            self.log("Depth cloud already saved at the 90deg step; loading it for cropping")
            try:
                self.depth_cloud_xyz = self._read_ply_xyz(os.path.join(self.out_dir, "depth_cloud.ply"))
            except Exception as exc:
                self.log(f"Reload depth cloud failed: {exc}")
            return
        msg = self.depth_cloud_snapshot
        if msg is None:
            self.log("No depth cloud snapshot taken at the photo pose; falling back to the latest message")
            deadline = time.monotonic() + fresh_timeout_s
            while time.monotonic() < deadline:
                ts = self.depth_cloud_ts
                if ts is not None and (time.monotonic() - ts) <= fresh_window_s:
                    break
                time.sleep(0.1)
            msg = self.depth_cloud_msg
        if msg is None:
            self.log(f"Depth point cloud not available (topic {self.depth_cloud_topic})")
            return
        try:
            xyz, rgb = self._pointcloud2_xyz_rgb(msg)
        except Exception as exc:
            self.log(f"Depth cloud parse failed: {exc}")
            return
        if xyz is None or len(xyz) == 0:
            self.log("Depth point cloud empty")
            return
        self.depth_cloud_xyz = xyz
        path = os.path.join(self.out_dir, "depth_cloud.ply")
        self._save_ply(path, xyz)
        self.log(f"Depth point cloud saved: {path} ({len(xyz)} points)")
        if rgb is not None:
            cpath = os.path.join(self.out_dir, "depth_cloud_colored.ply")
            self._save_ply_rgb(cpath, xyz, rgb)
            self.log(f"Colored depth cloud saved (camera-side, no extrinsics): {cpath}")
        else:
            self.log(f"Depth cloud has no rgb field (topic {self.depth_cloud_topic}); "
                     f"enable_colored_point_cloud:=true for a colored depth cloud")

    def process_depth_cloud(self):
        xyz = self.depth_cloud_xyz
        if xyz is None or len(xyz) == 0:
            self.log("Depth point cloud not available for cropping")
            return
        max_dist = self._max_dist
        if max_dist is not None:
            rng = np.linalg.norm(xyz, axis=1)
            xyz = xyz[(rng > self._min_dist) & (rng < max_dist)]
        path = os.path.join(self.out_dir, "depth_cloud_cropped.ply")
        self._save_ply(path, xyz)
        self.log(f"Depth point cloud cropped: {path} ({len(xyz)} points, max_dist={max_dist})")

    def _pointcloud2_xyz_rgb(self, msg: PointCloud2) -> tuple[np.ndarray, np.ndarray | None]:
        fields = {f.name: f.offset for f in msg.fields}
        if not all(name in fields for name in ("x", "y", "z")):
            raise ValueError("point cloud has no xyz fields")
        step = msg.point_step
        count = len(msg.data) // step
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=count * step).reshape(count, step)
        xyz = raw[:, :12].copy().view(np.float32)
        keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) > 0.0).any(axis=1)
        rgb = None
        if "rgb" in fields:
            off = fields["rgb"]
            rgb = raw[:, off:off + 3].copy()[keep]
        return xyz[keep], rgb

    def _save_ply_rgb(self, path: str, xyz: np.ndarray, rgb: np.ndarray):
        n = xyz.shape[0]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
        with open(path, "a") as f:
            np.savetxt(f, np.column_stack([xyz, rgb.astype(np.float64)]),
                       fmt="%.6f %.6f %.6f %.0f %.0f %.0f")

    def _pointcloud2_xyz(self, msg: PointCloud2) -> np.ndarray | None:
        try:
            import struct
            fields = {f.name: f.offset for f in msg.fields}
            if not all(name in fields for name in ('x', 'y', 'z')):
                return None
            point_step = msg.point_step
            data = msg.data
            num_points = len(data) // point_step
            pts = []
            for i in range(num_points):
                offset = i * point_step
                x = struct.unpack_from('f', data, offset + fields['x'])[0]
                y = struct.unpack_from('f', data, offset + fields['y'])[0]
                z = struct.unpack_from('f', data, offset + fields['z'])[0]
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and (x != 0.0 or y != 0.0 or z != 0.0):
                    pts.append([x, y, z])
            if not pts:
                return None
            return np.array(pts, dtype=np.float32)
        except Exception as exc:
            self.log(f"Depth cloud parse failed: {exc}")
            return None

    def open_lakibeam(self) -> bool:
        if self._lakibeam is not None:
            return True
        try:
            self._lakibeam = LakiBeamViewer(host_ip="0.0.0.0", port=self._lakibeam_port)
            self._lakibeam.connect()
        except Exception as exc:
            self.log(f"LakiBeam UDP bind failed: {exc}")
            self._lakibeam = None
            return False
        self.log(f"LakiBeam UDP bound 0.0.0.0:{self._lakibeam_port}")
        return True

    def close_lakibeam(self):
        if self._lakibeam is not None:
            self._lakibeam.close()
            self._lakibeam = None

    def _lakibeam_capture_loop(self):
        while self._lb_capturing and self._lakibeam is not None:
            scan = self._lakibeam.receive_scan()
            if scan:
                self._lb_frames.append((time.monotonic(), scan))

    def _start_lakibeam_capture(self):
        self._lb_frames.clear()
        if self._lakibeam is not None:
            self._lakibeam.clear_buffer()
        self._lb_capturing = True
        self._lb_thread = threading.Thread(target=self._lakibeam_capture_loop, daemon=True)
        self._lb_thread.start()

    def _stop_lakibeam_capture(self):
        self._lb_capturing = False
        if self._lb_thread is not None:
            self._lb_thread.join(timeout=3.0)
            self._lb_thread = None
        self.log(f"LakiBeam frames captured: {len(self._lb_frames)}")

    def _build_lakibeam_frame(self, scan, angle_deg, install, min_range, max_range):
        frame = scan_to_xy(scan)
        dist = np.linalg.norm(frame[:, :2], axis=1)
        frame = frame[(dist > min_range) & (dist <= max_range)]
        if frame.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        frame = frame @ install.lidar_tilt_matrix().T
        frame = install.mount_transform(frame)
        frame[:, 1] += install.offset_y_m
        frame[:, 2] += install.offset_z_m
        frame = install.to_world(frame)
        frame = frame @ rotation_matrix(install.turntable_axis, -angle_deg).T
        frame = frame @ install.tilt_matrix().T
        return np.asarray(frame, dtype=np.float64)

    def _merge_lakibeam(self, capture: ScanCapture):
        if not self._lb_frames or not capture.angle_samples:
            self.log("No LakiBeam data to merge")
            return
        try:
            install = InstallConfig.load(self._install_config_path)
        except Exception as exc:
            self.log(f"Install config load failed: {exc}")
            self.log("Merge aborted — 需要有效的 install yaml（不再回退到内置默认）")
            return
        self.log(f"Install config: {self._install_config_path}  mount={install.mount_axis}{install.mount_angle_deg}°  "
                 f"to_world x:{install.world_x} y:{install.world_y} z:{install.world_z}  "
                 f"offset y={install.offset_y_m} z={install.offset_z_m}")
        angle_times = np.array([s[0] for s in capture.angle_samples])
        angle_values = np.array([s[1] for s in capture.angle_samples])
        max_range = self._max_dist if self._max_dist is not None else 1.0e9
        min_range = self._min_dist if self._min_dist is not None else 0.0
        self._log_frame_angles(
            [(stamp, len(scan)) for stamp, scan in self._lb_frames], capture, "LakiBeam"
        )
        clouds = []
        for stamp, scan in self._lb_frames:
            angle_deg = np.interp(stamp, angle_times, angle_values)
            if not (capture.scan_start_deg <= angle_deg <= capture.scan_end_deg):
                continue
            cloud = self._build_lakibeam_frame(scan, angle_deg, install, min_range, max_range)
            if cloud.shape[0] > 0:
                clouds.append(cloud)
        if not clouds:
            self.log("No LakiBeam frames in scan range")
            return
        merged = np.concatenate(clouds, axis=0)
        out_path = os.path.join(capture.out_dir, "merged.ply")
        self._save_ply(out_path, merged)
        self.log(f"LakiBeam merged point cloud saved: {out_path}")
        self.log(f"LakiBeam merged frames: {len(clouds)}, total points: {len(merged)}")

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
        duration = FAST_MOVE_DURATION_S
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

        if self._lidar_kind == "lakibeam" and not self.open_lakibeam():
            self.log("LakiBeam UDP not available")
            return False

        if self.tt_status is not None and abs(self.tt_status.angle_deg - ready_deg) > 0.5:
            self.log(
                f"Not at ready position (current={self.tt_status.angle_deg:.2f}deg), "
                f"moving to {ready_deg}deg before scan..."
            )
            duration = FAST_MOVE_DURATION_S
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
        duration = FAST_MOVE_DURATION_S
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
        wait_s = max(60.0, scan_duration * 1.5 + 30.0)
        if not self.wait_for_turntable_idle(target_deg=scan_end_deg, timeout_s=wait_s):
            self.log(f"Scan move timeout after {wait_s:.0f}s (scan_duration={scan_duration:.1f}s)")
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

        duration = FAST_MOVE_DURATION_S
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
        if self._lidar_kind == "lakibeam":
            self._start_lakibeam_capture()
        self._capturing = True
        self.log("Capture started")

    def _stop_capture(self):
        self._capturing = False
        if self._lidar_kind == "lakibeam":
            self._stop_lakibeam_capture()
        self.log("Capture stopped")

    def _process_frames_and_merge(self, capture: ScanCapture):
        self.save_depth_cloud()
        if self._lidar_kind == "lakibeam":
            self._merge_lakibeam(capture)
            self.process_depth_cloud()
            photo_angle = (capture.scan_start_deg + capture.scan_end_deg) / 2.0
            self._colorize_merged(capture.out_dir, photo_angle)
            self._detect_board(capture.out_dir)
            return
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
                    self._save_ply(ply_path, points[:, :3])
            self.log(f"Processed {len(captured_frames)} frames")
        self.stitch_and_save_merged(captured_frames, capture)
        self.process_depth_cloud()

    def _log_frame_angles(
        self,
        frames: list[tuple[float, int]],
        capture: ScanCapture,
        label: str,
    ) -> None:
        """把采集到的每帧（相对时间、点数、插值角度、是否在扫描范围内）打到日志。"""
        if not frames:
            self.log(f"{label}: no frames captured")
            return
        if not capture.angle_samples:
            self.log(f"{label}: {len(frames)} frames captured, but no angle samples")
            return
        angle_times = np.array([s[0] for s in capture.angle_samples])
        angle_values = np.array([s[1] for s in capture.angle_samples])
        t0 = frames[0][0]
        self.log(
            f"{label} frames ({len(frames)}), scan range "
            f"[{capture.scan_start_deg:.1f}, {capture.scan_end_deg:.1f}] deg:"
        )
        for i, (stamp, n_pts) in enumerate(frames):
            angle_deg = float(np.interp(stamp, angle_times, angle_values))
            skip = "" if capture.scan_start_deg <= angle_deg <= capture.scan_end_deg else "  SKIP: out of range"
            self.log(f"  {i:03d}  t+{stamp - t0:6.3f}s  {n_pts:6d} pts  angle={angle_deg:8.2f} deg{skip}")

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
        keep = list(MIDDLE_12_RINGS)
        max_dist = self._max_dist
        self.log(f"Merging rings {keep[0]}-{keep[-1]}, max_dist={max_dist}")
        self._log_frame_angles(
            [(stamp, len(data)) for stamp, data in captured_frames], capture, "Fairy"
        )

        all_points = []
        for frame_stamp, frame_data in captured_frames:
            angle_deg = np.interp(frame_stamp, angle_times, angle_values)
            if not (capture.scan_start_deg <= angle_deg <= capture.scan_end_deg):
                continue
            xyz = frame_data[:, :3]
            rings = frame_data[:, 3].astype(np.int16)
            valid = np.isfinite(xyz).all(axis=1)
            frame_xyz = xyz[valid & np.isin(rings, keep)]
            if max_dist is not None:
                rng = np.linalg.norm(frame_xyz, axis=1)
                frame_xyz = frame_xyz[(rng > self._min_dist) & (rng < max_dist)]
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
        self._detect_board(capture.out_dir)

    def _detect_board(self, out_dir: str) -> None:
        """每次扫描合并后跑 segment_board 找板, 产出 board_rect.ply + overlay。
        独立于外参(无上色配置也跑); 失败只记日志, 不影响扫描流程。"""
        merged = os.path.join(out_dir, "merged.ply")
        if not os.path.exists(merged):
            return
        try:
            from segment_board import DIST_THR, extract_rect_plane, write_ply
            P = self._read_ply_xyz(merged)
            mask, info = extract_rect_plane(P)
            if mask is None:
                self.log(f"Board: not found in {os.path.basename(out_dir)}")
                return
            write_ply(os.path.join(out_dir, "board_rect.ply"), P[mask])
            band_full = info.get("band_mask")
            if band_full is None:
                band_full = np.abs(P @ info["n"] - info["d"]) < DIST_THR
            cols = np.full((len(P), 3), 120, np.uint8)
            cols[band_full & ~mask] = (30, 30, 255)
            cols[mask] = (255, 30, 30)
            write_ply(os.path.join(out_dir, "board_rect_overlay.ply"), P, cols)
            ratio = info["n_sel"] / max(int(info.get("band_cc_n", 1)), 1)
            self.log(f"Board: {os.path.basename(out_dir)} selected={info['n_sel']} "
                     f"band_ratio={ratio:.2f}")
        except Exception as exc:
            self.log(f"Board detection failed: {exc}")

    def _colorize_merged(self, out_dir: str, photo_angle_deg: float):
        extrinsics = os.path.join(os.getcwd(), "config", "camera_extrinsics.yaml")
        camera_info = os.path.join(os.getcwd(), "config", "camera_info.yaml")
        if not (os.path.exists(extrinsics) and os.path.exists(camera_info)):
            self.log("Colorize skipped: config/camera_extrinsics.yaml or camera_info.yaml missing")
            return
        try:
            import yaml as _yaml
            _ex = _yaml.safe_load(open(extrinsics))["lidar_to_camera"]
            self.log(f"Colorize extrinsics: {extrinsics} "
                     f"t={[round(float(v), 5) for v in _ex['translation']]}")
        except Exception as exc:
            self.log(f"Colorize extrinsics read failed: {exc}")
        try:
            from colorize_pointcloud import colorize
            out, total, colored = colorize(out_dir, extrinsics, camera_info, photo_angle_deg)
            self.log(f"Colored point cloud saved: {out} ({colored}/{total} points)")
        except Exception as exc:
            self.log(f"Colorize failed: {exc}")

    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray | None:
        try:
            import struct
            fields = {f.name: f.offset for f in msg.fields}
            if not all(name in fields for name in ('x', 'y', 'z')):
                return None
            width = msg.width or 1
            point_step = msg.point_step
            data = msg.data
            num_points = len(data) // point_step
            out = np.empty((num_points, 4), dtype=np.float32)
            for i in range(num_points):
                offset = i * point_step
                out[i, 0] = struct.unpack_from('f', data, offset + fields['x'])[0]
                out[i, 1] = struct.unpack_from('f', data, offset + fields['y'])[0]
                out[i, 2] = struct.unpack_from('f', data, offset + fields['z'])[0]
                out[i, 3] = i % width
            return out
        except Exception as exc:
            self.log(f"PointCloud2 parse failed: {exc}")
            return None

    def _read_ply_xyz(self, path: str) -> np.ndarray:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if line.strip() == "end_header":
                    break
            else:
                raise ValueError(f"{path}: no end_header")
        data = np.loadtxt(path, skiprows=i, dtype=np.float64)
        return data[:, :3].astype(np.float32)

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

        if self._no_fairy or self._lidar_kind == "lakibeam":
            checks.append("LiDAR: skipped")
        else:
            if self._fairy_msg_count > 0:
                checks.append(f"LiDAR: has data ({self._fairy_msg_count} frames)")
            else:
                checks.append("LiDAR: no data")
                ready = False

        return ready, "; ".join(checks)

    def launch_sensor_env(self, timeout_s: float = 120.0, use_fairy: bool = True,
                          turntable_port: str | None = None) -> bool:
        if turntable_port:
            self._turntable_port = turntable_port
        proc = self._launch_proc
        if proc is None or proc.poll() is not None:
            self.reset_usb_cameras()
            time.sleep(2.0)
            cmd = SENSOR_ENV_LAUNCH.split() + [
                f"use_fairy:={'true' if use_fairy else 'false'}",
                f"turntable_port:={self._turntable_port}",
            ]
            self.log(f"Starting background services: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
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

        self.font_name = tkfont.nametofont("TkDefaultFont").actual("family")
        self.font_large = (self.font_name, 14)
        self.font_normal = (self.font_name, 11)

        self._create_widgets()
        node._lidar_kind = self._selected_lidar_kind()
        node.set_log_callback(self._log)
        self._set_buttons(tk.DISABLED, tk.DISABLED, tk.NORMAL)
        self._update_status(False, "Press 'Restart services' to start")

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

        dist_frame = tk.Frame(self.root)
        dist_frame.pack(pady=5)

        tk.Label(dist_frame, text="Range (min, max):", font=self.font_normal).pack(side=tk.LEFT)
        self.dist_var = tk.StringVar(value=f"{self.node._min_dist:g}, {self.node._max_dist:g}")
        self.dist_entry = tk.Entry(dist_frame, textvariable=self.dist_var, width=14, font=self.font_normal)
        self.dist_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(dist_frame, text="m", font=self.font_normal).pack(side=tk.LEFT)

        speed_frame = tk.Frame(self.root)
        speed_frame.pack(pady=5)

        tk.Label(speed_frame, text="Scan speed:", font=self.font_normal).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value=f"{self.sweep_speed:g}")
        self.speed_entry = tk.Entry(speed_frame, textvariable=self.speed_var, width=8, font=self.font_normal)
        self.speed_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(speed_frame, text="deg/s", font=self.font_normal).pack(side=tk.LEFT)

        lidar_frame = tk.Frame(self.root)
        lidar_frame.pack(pady=5)

        tk.Label(lidar_frame, text="LiDAR:", font=self.font_normal).pack(side=tk.LEFT)
        self.lidar_var = tk.StringVar(value="LakiBeam (UDP)")
        self.lidar_combo = ttk.Combobox(
            lidar_frame, textvariable=self.lidar_var, state="readonly", width=16,
            values=["Fairy (ROS)", "LakiBeam (UDP)"], font=self.font_normal,
        )
        self.lidar_combo.pack(side=tk.LEFT, padx=5)
        self.restart_btn = tk.Button(
            lidar_frame, text="Restart services", font=self.font_normal,
            command=self._on_restart_services, state=tk.DISABLED,
        )
        self.restart_btn.pack(side=tk.LEFT, padx=5)

        port_frame = tk.Frame(self.root)
        port_frame.pack(pady=5)

        tk.Label(port_frame, text="Turntable port:", font=self.font_normal).pack(side=tk.LEFT)
        ports = list_serial_ports()
        if ports:
            initial_port = next((v for v in ports if port_from_label(v) == self.node._turntable_port), ports[0])
        else:
            initial_port = self.node._turntable_port
        self.port_var = tk.StringVar(value=initial_port)
        self.port_combo = ttk.Combobox(
            port_frame, textvariable=self.port_var, width=30,
            values=ports, font=self.font_normal,
        )
        self.port_combo.pack(side=tk.LEFT, padx=5)
        self.port_combo.bind("<<ComboboxSelected>>", self._on_port_selected)
        self.port_combo.bind("<FocusOut>", self._on_port_selected)
        tk.Button(
            port_frame, text="Refresh", font=self.font_normal,
            command=self._on_refresh_ports,
        ).pack(side=tk.LEFT)

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

        if not ports:
            self._log("No USB serial ports found. If the adapter was plugged in after the "
                      "container started, restart the dev container (its /dev is a private "
                      "tmpfs and misses re-enumerated devices).")

    def _update_status(self, ready: bool, text: str):
        color = "green" if ready else "red"
        self.status_canvas.itemconfig(self.status_dot, fill=color)
        self.status_label.config(text=text)

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

    def _get_range(self) -> tuple[float, float] | None:
        parts = [p.strip() for p in self.dist_var.get().split(",")]
        try:
            if len(parts) != 2:
                raise ValueError("格式为 min,max（半角逗号分隔）")
            low, high = float(parts[0]), float(parts[1])
            if low < 0 or high <= low:
                raise ValueError("需满足 0 <= min < max")
            return low, high
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid distance range: {e}")
            return None

    def _get_sweep_speed(self) -> float | None:
        try:
            value = float(self.speed_var.get())
            if value <= 0:
                raise ValueError("must be positive")
            return value
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid scan speed: {e}")
            return None

    def _run_in_thread(self, func):
        def wrapper():
            try:
                func()
            except Exception as e:
                self.root.after(0, lambda: self._log(f"Error: {e}"))
        threading.Thread(target=wrapper, daemon=True).start()

    def _set_buttons(self, reset_state, scan_state, restart_state=None):
        self.reset_btn.config(state=reset_state)
        self.scan_btn.config(state=scan_state)
        self.restart_btn.config(state=scan_state if restart_state is None else restart_state)

    def _on_reset(self):
        speed = self._get_sweep_speed()
        if speed is None:
            return
        self.sweep_speed = speed
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
            self._set_buttons(tk.NORMAL, tk.DISABLED, tk.NORMAL)
            messagebox.showerror("Error", "Reset failed")

    def _selected_lidar_kind(self) -> str:
        return "lakibeam" if self.lidar_var.get().startswith("LakiBeam") else "fairy"

    def _selected_turntable_port(self) -> str:
        return port_from_label(self.port_var.get()) or self.node._turntable_port

    def _on_port_selected(self, _event=None):
        port = self._selected_turntable_port()
        if port == self.node._turntable_port:
            return
        self.node._turntable_port = port
        running = self.node._launch_proc is not None and self.node._launch_proc.poll() is None
        suffix = " (takes effect on next Restart services)" if running else ""
        self._log(f"Turntable port set to {port}{suffix}")

    def _on_refresh_ports(self):
        values = list_serial_ports()
        self.port_combo["values"] = values
        current = self._selected_turntable_port()
        match = next((v for v in values if port_from_label(v) == current), None)
        if match:
            self.port_var.set(match)
        elif values:
            self.port_var.set(values[0])
        self.node._turntable_port = self._selected_turntable_port()
        if not values:
            self._log("No USB serial ports found. If the adapter was plugged in after the "
                      "container started, restart the dev container (its /dev is a private "
                      "tmpfs and misses re-enumerated devices).")

    def _on_restart_services(self):
        kind = self._selected_lidar_kind()
        self._set_buttons(tk.DISABLED, tk.DISABLED)
        self._run_in_thread(lambda: self._do_restart_services(kind))

    def _do_restart_services(self, kind: str):
        self.node._lidar_kind = kind
        self.node._turntable_port = self._selected_turntable_port()
        use_fairy = kind == "fairy"
        ok = self.node.restart_sensor_env(use_fairy=use_fairy)
        if ok:
            _, detail = self.node.check_ready(self.color_topic, self.depth_topic, self.fairy_topic)
            self.root.after(0, lambda d=detail: self._log(d))
        self.root.after(0, lambda: self._after_restart_services(ok, use_fairy))

    def _after_restart_services(self, ok: bool, use_fairy: bool):
        self._set_buttons(tk.NORMAL, tk.DISABLED, tk.NORMAL)
        if ok:
            self._update_status(True, "READY")
            self._log(f"Services restarted (fairy={'on' if use_fairy else 'off'}, "
                      f"port={self.node._turntable_port})")
        else:
            messagebox.showerror("Error", "Restart services failed")

    def _on_scan(self):
        scan_range = self._get_scan_range()
        if scan_range is None:
            return
        dist_range = self._get_range()
        if dist_range is None:
            return
        speed = self._get_sweep_speed()
        if speed is None:
            return
        self.sweep_speed = speed
        self.node._min_dist, self.node._max_dist = dist_range
        self.node._lidar_kind = self._selected_lidar_kind()
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
    parser.add_argument("--depth-cloud-topic", default="/camera/depth_registered/points", help="Depth point cloud topic")
    parser.add_argument("--fairy-topic", default="/rslidar_points", help="LiDAR point cloud topic")
    parser.add_argument("--ready-deg", type=float, default=90.0, help="Ready angle")
    parser.add_argument("--sweep-speed", type=float, default=40.0, help="Sweep speed")
    parser.add_argument("--no-fairy", action="store_true", help="Skip LiDAR capture")
    parser.add_argument("--rotation-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Merge rotation axis")
    parser.add_argument("--lakibeam-port", type=int, default=2368, help="LakiBeam UDP port")
    parser.add_argument("--install-config", default="config/install_side_mount.yaml",
                        help="LakiBeam install config YAML")
    parser.add_argument("--min-dist", type=float, default=MIN_RANGE_M,
                        help="Near range cutoff in meters (merged points closer than this are dropped)")
    parser.add_argument("--max-dist", type=float, default=3.0,
                        help="Far range cutoff in meters")
    args = parser.parse_args()

    config = {
        "no_fairy": args.no_fairy,
        "rotation_axis": list(args.rotation_axis),
        "depth_cloud_topic": args.depth_cloud_topic,
        "lakibeam_port": args.lakibeam_port,
        "install_config": args.install_config,
        "min_dist": args.min_dist,
        "max_dist": args.max_dist,
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
