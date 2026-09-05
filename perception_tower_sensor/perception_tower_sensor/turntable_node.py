"""Turntable control ROS2 node.

Serial protocol (115200 8N1, #...! delimiters):
    MOVE : #000P{pos}T{time_ms}!
    READ : #000PRAD!           -> #000P{pos}!
    STOP : #000PDST!           -> #OK!
    RST  : #000PRST!           -> #OK!

Publishes /turntable/status at 50 Hz.
Provides /turntable/command service.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from perception_tower_sensor_interfaces.msg import TurntableStatus
from perception_tower_sensor_interfaces.srv import TurntableCommand


# --- Protocol parser ---

_OK_EVENT = ("ok",)
_POSITION_RE = re.compile(rb"^(\d{3})P(\d+)$")


class ProtocolParser:
    def __init__(self, servo_id: int = 0):
        self._id = servo_id
        self._buf = bytearray()
        self._id_bytes = f"{servo_id:03d}".encode()

    def feed(self, data: bytes) -> List[tuple]:
        self._buf.extend(data)
        events: List[tuple] = []
        while True:
            start = self._buf.find(b"#")
            if start < 0:
                self._buf.clear()
                break
            if start > 0:
                del self._buf[:start]
            end = self._buf.find(b"!")
            if end < 0:
                break
            chunk = bytes(self._buf[1:end])
            del self._buf[: end + 1]
            if chunk == b"OK":
                events.append(_OK_EVENT)
            else:
                m = _POSITION_RE.match(chunk)
                if m and m.group(1) == self._id_bytes:
                    events.append(("pos", int(m.group(2))))
        return events


# --- Servo errors ---

class ServoError(RuntimeError):
    pass


# --- Serial client ---

class ServoClient:
    def __init__(self, port: str, baud: int = 115200, servo_id: int = 0,
                 pos_origin: int = 500, deg_per_pos: float = 0.02):
        self._port = port
        self._baud = baud
        self._servo_id = servo_id
        self._origin = pos_origin
        self._dpp = deg_per_pos
        self._ser = None
        self._parser = ProtocolParser(servo_id)
        self._reply_q: "queue.Queue[tuple]" = queue.SimpleQueue()
        self._write_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

    def open(self):
        import serial
        if not self._port:
            raise ServoError("serial_port not configured")
        self._ser = serial.Serial(self._port, self._baud, timeout=0.05)
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def close(self):
        self._running = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.5)

    def _read_loop(self):
        while self._running:
            try:
                data = self._ser.read(256)
            except Exception:
                break
            if data:
                events = self._parser.feed(data)
                for ev in events:
                    self._reply_q.put(ev)

    def _send(self, payload: bytes):
        with self._write_lock:
            if self._ser is None:
                raise ServoError("serial not open")
            self._ser.write(payload)

    def _wait_event(self, kinds: tuple, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise ServoError(f"timeout waiting for {kinds}")
            try:
                ev = self._reply_q.get(timeout=min(remain, 0.1))
            except queue.Empty:
                continue
            if ev[0] in kinds:
                return ev

    def _flush_replies(self):
        while True:
            try:
                self._reply_q.get_nowait()
            except queue.Empty:
                break

    def move_to(self, pos: int, time_ms: int):
        self._send(f"#{self._servo_id:03d}P{pos}T{time_ms}!".encode())

    def stop(self):
        self._send(f"#{self._servo_id:03d}PDST!".encode())
        self._wait_event(("ok",), 0.5)

    def read_position(self, timeout_s: float = 0.2) -> int:
        self._flush_replies()
        self._send(f"#{self._servo_id:03d}PRAD!".encode())
        ev = self._wait_event(("pos",), timeout_s)
        return int(ev[1])

    def reset(self, timeout_s: float = 30.0):
        self._send(f"#{self._servo_id:03d}PRST!".encode())
        self._wait_event(("ok",), timeout_s)

    def pos_to_deg(self, pos: int) -> float:
        return (pos - self._origin) * self._dpp

    def deg_to_pos(self, deg: float) -> int:
        return int(round(self._origin + deg / self._dpp))


# --- ROS2 node ---

class TurntableNode(Node):
    def __init__(self):
        super().__init__("turntable_node")
        self._declare_params()
        self._load_params()

        self._servo = ServoClient(
            port=self._port,
            baud=self._baud,
            pos_origin=self._origin,
            deg_per_pos=self._dpp,
        )
        try:
            self._servo.open()
            self.get_logger().info(f"serial opened: {self._port}")
        except Exception as exc:
            self.get_logger().error(f"failed to open serial: {exc}")
            raise

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._status_pub = self.create_publisher(TurntableStatus, "/turntable/status", qos)
        self._srv = self.create_service(TurntableCommand, "/turntable/command", self._on_command)

        self._state = TurntableStatus.STATE_IDLE
        self._last_pos = self._origin
        self._lock = threading.Lock()

        pub_period = 1.0 / self._pub_hz
        self._pub_timer = self.create_timer(pub_period, self._publish_status)

        poll_period = 1.0 / self._poll_hz
        self._poll_thread = threading.Thread(target=self._poll_loop, args=(poll_period,), daemon=True)
        self._poll_thread.start()

    def _declare_params(self):
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("serial_baud", 115200)
        self.declare_parameter("poll_hz", 100.0)
        self.declare_parameter("pub_hz", 50.0)
        self.declare_parameter("pos_origin", 500)
        self.declare_parameter("deg_per_pos", 0.02)
        self.declare_parameter("angle_sign", 1)
        self.declare_parameter("home_timeout_s", 30.0)

    def _load_params(self):
        self._port = self.get_parameter("serial_port").value
        self._baud = self.get_parameter("serial_baud").value
        self._poll_hz = self.get_parameter("poll_hz").value
        self._pub_hz = self.get_parameter("pub_hz").value
        self._origin = self.get_parameter("pos_origin").value
        self._dpp = self.get_parameter("deg_per_pos").value
        self._angle_sign = self.get_parameter("angle_sign").value
        self._home_timeout = self.get_parameter("home_timeout_s").value

    def _poll_loop(self, period: float):
        while rclpy.ok():
            t0 = time.monotonic()
            try:
                pos = self._servo.read_position(timeout_s=period * 2.0)
                with self._lock:
                    self._last_pos = pos
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _publish_status(self):
        msg = TurntableStatus()
        with self._lock:
            pos = self._last_pos
        msg.position = float(pos)
        msg.angle_deg = self._servo.pos_to_deg(pos)
        msg.state = self._state
        self._status_pub.publish(msg)

    def _on_command(self, request, response):
        cmd = request.command
        if cmd == TurntableCommand.Request.CMD_HOME:
            self._state = TurntableStatus.STATE_HOMING
            try:
                self._servo.reset(timeout_s=self._home_timeout)
                target_deg = request.target_deg if request.target_deg else 90.0
                target = self._servo.deg_to_pos(target_deg)
                time_ms = max(200, int(abs(target_deg) / 40.0 * 1000))
                self._servo.move_to(target, time_ms)
                self._state = TurntableStatus.STATE_IDLE
                response.success = True
                response.message = "homed"
            except Exception as exc:
                self._state = TurntableStatus.STATE_ERROR
                response.success = False
                response.message = f"home failed: {exc}"

        elif cmd == TurntableCommand.Request.CMD_MOVE:
            self._state = TurntableStatus.STATE_MOVING
            try:
                pos = self._servo.deg_to_pos(request.target_deg)
                time_ms = max(200, int(request.duration_s * 1000)) if request.duration_s > 0 else 2000
                self._servo.move_to(pos, time_ms)
                response.success = True
                response.message = f"moving to {request.target_deg:.1f} deg"
            except Exception as exc:
                self._state = TurntableStatus.STATE_ERROR
                response.success = False
                response.message = f"move failed: {exc}"

        elif cmd == TurntableCommand.Request.CMD_STOP:
            try:
                self._servo.stop()
                self._state = TurntableStatus.STATE_IDLE
                response.success = True
                response.message = "stopped"
            except Exception as exc:
                self._state = TurntableStatus.STATE_ERROR
                response.success = False
                response.message = f"stop failed: {exc}"

        else:
            response.success = False
            response.message = f"unknown command {cmd}"

        return response

    def destroy_node(self):
        try:
            self._servo.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TurntableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
