"""Turntable control ROS2 node.

Serial protocol (115200 8N1):
    Text commands (host -> device):
        MOVE : #000P{pos}T{time_ms}!
        READ : #000PRAD!
        STOP : #000PDST!           -> #OK!
        RST  : #000PRST!           -> #OK!
        STR  : #000PSTR{interval_ms}!   start auto report
        STP  : #000PSTP!                stop auto report
    Binary position frame (device -> host):
        AA 55 06 01 BATCH POS_H POS_M POS_L DONE CRC8

Publishes /turntable/status driven by automatic position reports.
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
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Header


# --- Protocol constants ---

_OK_EVENT = ("ok",)
_POSITION_RE = re.compile(rb"^(\d{3})P(\d+)$")

_BINARY_FRAME_LEN = 10
_BINARY_HEAD = b"\xAA\x55"
_BINARY_LEN = 0x06
_BINARY_TYPE = 0x01


def _crc8(data: bytes) -> int:
    """Dallas/Maxim CRC8 (poly 0x31, init 0x00)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class ProtocolParser:
    def __init__(self, servo_id: int = 0):
        self._id = servo_id
        self._buf = bytearray()
        self._id_bytes = f"{servo_id:03d}".encode()

    def feed(self, data: bytes) -> List[tuple]:
        self._buf.extend(data)
        events: List[tuple] = []
        while True:
            # Skip garbage until we see a known frame start.
            while len(self._buf) >= 2:
                if self._buf[0] == _BINARY_HEAD[0] and self._buf[1] == _BINARY_HEAD[1]:
                    break
                if self._buf[0] == ord("#"):
                    break
                del self._buf[0]

            if not self._buf:
                break

            # Binary frame.
            if self._buf[0] == _BINARY_HEAD[0]:
                if len(self._buf) < _BINARY_FRAME_LEN:
                    break
                frame = bytes(self._buf[:_BINARY_FRAME_LEN])
                del self._buf[:_BINARY_FRAME_LEN]

                if frame[2] != _BINARY_LEN or frame[3] != _BINARY_TYPE:
                    print(f"[parser] unexpected binary frame len/type: {frame[2]:02X} {frame[3]:02X}", flush=True)
                    continue

                crc = _crc8(frame[2:9])
                if crc != frame[9]:
                    print(f"[parser] crc mismatch: calc={crc:02X} rx={frame[9]:02X}", flush=True)
                    continue

                pos = (frame[5] << 16) | (frame[6] << 8) | frame[7]
                batch = frame[4]
                done = bool(frame[8])
                events.append(("pos", pos, batch, done))
                continue

            # Text frame.
            if self._buf[0] == ord("#"):
                end = self._buf.find(b"!")
                if end < 0:
                    break
                chunk = bytes(self._buf[1:end]).strip()
                del self._buf[: end + 1]
                # Drop trailing \r\n between frames.
                while self._buf[:1] in (b"\r", b"\n"):
                    del self._buf[0]

                if chunk.upper() == b"OK":
                    events.append(_OK_EVENT)
                else:
                    m = _POSITION_RE.match(chunk)
                    if m and m.group(1) == self._id_bytes:
                        events.append(("pos", int(m.group(2)), 0, True))
                continue

            # Should never get here.
            break
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

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _read_loop(self):
        while self._running:
            try:
                data = self._ser.read(256)
            except Exception as exc:
                print(f"[servo] read loop error: {exc}", flush=True)
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                break
            if data:
                events = self._parser.feed(data)
                for ev in events:
                    self._reply_q.put(ev)

    def _send(self, payload: bytes):
        with self._write_lock:
            if self._ser is None:
                raise ServoError("serial not open")
            # The protocol requires commands to end with \r\n.
            if not payload.endswith(b"\r\n"):
                payload = payload + b"\r\n"
            print(f"[servo] tx {payload!r}", flush=True)
            try:
                self._ser.write(payload)
            except Exception as exc:
                print(f"[servo] write failed: {exc}", flush=True)
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                raise ServoError(f"write failed: {exc}") from exc

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
            print(f"[servo] rx event {ev}", flush=True)
            if ev[0] in kinds:
                return ev

    def _flush_replies(self):
        count = 0
        while True:
            try:
                self._reply_q.get_nowait()
                count += 1
            except queue.Empty:
                break
        if count:
            print(f"[servo] flushed {count} stale events", flush=True)

    def move_to(self, pos: int, time_ms: int):
        cmd = f"#{self._servo_id:03d}P{pos}T{time_ms}!"
        print(f"[servo] MOVE pos={pos} time_ms={time_ms} -> {cmd}", flush=True)
        self._send(cmd.encode())

    def start_auto_report(self, interval_ms: int = 20):
        interval_ms = max(10, int(interval_ms))
        cmd = f"#{self._servo_id:03d}PSTR{interval_ms}!"
        print(f"[servo] START_AUTO_REPORT interval={interval_ms}ms -> {cmd}", flush=True)
        self._send(cmd.encode())

    def stop_auto_report(self):
        cmd = f"#{self._servo_id:03d}PSTP!"
        print(f"[servo] STOP_AUTO_REPORT -> {cmd}", flush=True)
        self._send(cmd.encode())

    def stop(self):
        self._flush_replies()
        self._send(f"#{self._servo_id:03d}PDST!".encode())
        try:
            self._wait_event(("ok",), 0.5)
        except ServoError:
            print("[servo] stop did not get OK, continuing", flush=True)

    def read_position(self, timeout_s: float = 0.2) -> tuple:
        self._flush_replies()
        self._send(f"#{self._servo_id:03d}PRAD!".encode())
        ev = self._wait_event(("pos",), timeout_s)
        return int(ev[1]), int(ev[2]), bool(ev[3])

    def reset(self, timeout_s: float = 30.0):
        self._flush_replies()
        for attempt in range(1, 4):
            self._send(f"#{self._servo_id:03d}PRST!".encode())
            try:
                self._wait_event(("ok",), 2.0)
                print(f"[servo] reset acknowledged on attempt {attempt}", flush=True)
                return
            except ServoError:
                print(f"[servo] reset attempt {attempt} timed out waiting for OK", flush=True)
        # Some controllers do not reply to RST; verify the device is alive by reading position.
        try:
            pos, batch, done = self.read_position(timeout_s=1.0)
            print(f"[servo] reset got no OK but device is alive at pos={pos}", flush=True)
        except ServoError as exc:
            raise ServoError(f"reset failed: no OK and cannot read position: {exc}") from exc

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
        self._state = TurntableStatus.STATE_IDLE
        try:
            self._servo.open()
            self.get_logger().info(f"serial opened: {self._port}")
            try:
                self._servo.start_auto_report(self._auto_report_ms)
            except Exception as exc:
                self.get_logger().warning(f"failed to start auto report: {exc}")
        except Exception as exc:
            self.get_logger().error(f"failed to open serial: {exc}; turntable commands will be unavailable")
            self._state = TurntableStatus.STATE_ERROR

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._status_pub = self.create_publisher(TurntableStatus, "/turntable/status", qos)
        self._srv = self.create_service(TurntableCommand, "/turntable/command", self._on_command)

        self._last_pos = self._origin
        self._last_batch = 0
        self._last_done = True
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        pub_period = 1.0 / self._pub_hz
        self._pub_timer = self.create_timer(pub_period, self._publish_status)

        # Use a dedicated thread to consume serial data. If the user explicitly sets
        # poll_hz, use that rate; otherwise, when auto-report is enabled, drain the
        # reply queue at 100 Hz so position updates are not delayed.
        if self._poll_hz > 0.0:
            loop_hz = self._poll_hz
        elif self._auto_report_ms > 0:
            loop_hz = 100.0
        else:
            loop_hz = 0.0

        if loop_hz > 0.0:
            poll_period = 1.0 / loop_hz
            self._poll_thread = threading.Thread(target=self._poll_loop, args=(poll_period,), daemon=True)
            self._poll_thread.start()
        else:
            self._poll_thread = None

        self.add_on_set_parameters_callback(self._on_param_change)

    def _on_param_change(self, params):
        restart_poll = False
        for param in params:
            if param.name == "poll_hz":
                self._poll_hz = param.value
                restart_poll = True
            elif param.name == "auto_report_ms":
                self._auto_report_ms = param.value
                if self._servo.is_open and self._auto_report_ms > 0:
                    try:
                        self._servo.start_auto_report(self._auto_report_ms)
                        self.get_logger().info(f"updated auto report interval to {self._auto_report_ms} ms")
                    except Exception as exc:
                        self.get_logger().warning(f"failed to update auto report: {exc}")
                restart_poll = True

        if restart_poll:
            if self._poll_thread is not None:
                self._shutdown.set()
                self._poll_thread.join(timeout=2.0)
                self._poll_thread = None
                self._shutdown.clear()

            if self._poll_hz > 0.0:
                loop_hz = self._poll_hz
            elif self._auto_report_ms > 0:
                loop_hz = 100.0
            else:
                loop_hz = 0.0

            if loop_hz > 0.0:
                poll_period = 1.0 / loop_hz
                self.get_logger().info(f"restarting serial consume loop at {loop_hz} Hz")
                self._poll_thread = threading.Thread(target=self._poll_loop, args=(poll_period,), daemon=True)
                self._poll_thread.start()

        return SetParametersResult(successful=True)

    def _declare_params(self):
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("serial_baud", 115200)
        self.declare_parameter("poll_hz", 0.0)
        self.declare_parameter("pub_hz", 50.0)
        self.declare_parameter("auto_report_ms", 20)
        self.declare_parameter("pos_origin", 500)
        self.declare_parameter("deg_per_pos", 0.02)
        self.declare_parameter("angle_sign", 1)
        self.declare_parameter("home_timeout_s", 30.0)

    def _load_params(self):
        self._port = self.get_parameter("serial_port").value
        self._baud = self.get_parameter("serial_baud").value
        self._poll_hz = self.get_parameter("poll_hz").value
        self._pub_hz = self.get_parameter("pub_hz").value
        self._auto_report_ms = self.get_parameter("auto_report_ms").value
        self._origin = self.get_parameter("pos_origin").value
        self._dpp = self.get_parameter("deg_per_pos").value
        self._angle_sign = self.get_parameter("angle_sign").value
        self._home_timeout = self.get_parameter("home_timeout_s").value

    def _poll_loop(self, period: float):
        reconnect_delay = 1.0
        while rclpy.ok() and not self._shutdown.is_set():
            if not self._servo.is_open:
                try:
                    self._servo.open()
                    self.get_logger().info(f"serial reopened: {self._port}")
                    self._state = TurntableStatus.STATE_IDLE
                    reconnect_delay = 1.0
                    try:
                        self._servo.start_auto_report(self._auto_report_ms)
                    except Exception as exc:
                        self.get_logger().warning(f"failed to restart auto report: {exc}")
                except Exception as exc:
                    self.get_logger().warning(f"serial reopen failed: {exc}; retry in {reconnect_delay:.1f}s")
                    self._state = TurntableStatus.STATE_ERROR
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay + 1.0, 10.0)
                    continue
            t0 = time.monotonic()
            try:
                if self._auto_report_ms > 0:
                    latest = None
                    while True:
                        try:
                            ev = self._servo._reply_q.get_nowait()
                        except queue.Empty:
                            break
                        if ev[0] == "pos":
                            latest = ev
                    if latest is not None:
                        with self._lock:
                            self._last_pos = int(latest[1])
                            self._last_batch = int(latest[2])
                            self._last_done = bool(latest[3])
                elif self._poll_hz > 0.0:
                    pos, batch, done = self._servo.read_position(timeout_s=period * 2.0)
                    with self._lock:
                        self._last_pos = pos
                        self._last_batch = batch
                        self._last_done = done
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            if elapsed < period and not self._shutdown.is_set():
                time.sleep(period - elapsed)

    def _publish_status(self):
        msg = TurntableStatus()
        with self._lock:
            pos = self._last_pos
            batch = self._last_batch
            done = self._last_done
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "turntable"
        msg.position = float(pos)
        msg.angle_deg = self._servo.pos_to_deg(pos) * self._angle_sign
        msg.state = self._state
        msg.batch = batch
        msg.done = done
        self._status_pub.publish(msg)

    def _on_command(self, request, response):
        cmd = request.command
        self.get_logger().info(
            f"received turntable command: cmd={cmd}, target_deg={request.target_deg:.2f}, duration_s={request.duration_s:.2f}"
        )
        if not self._servo.is_open:
            response.success = False
            response.message = "serial not open"
            self.get_logger().error(response.message)
            return response
        if cmd == TurntableCommand.Request.CMD_HOME:
            self.get_logger().info("executing HOME command")
            self._state = TurntableStatus.STATE_HOMING
            try:
                self._servo.reset(timeout_s=self._home_timeout)
                if self._auto_report_ms > 0:
                    self._servo.start_auto_report(self._auto_report_ms)
                target_deg = request.target_deg if request.target_deg else 90.0
                target = self._servo.deg_to_pos(target_deg)
                time_ms = max(200, int(abs(target_deg) / 40.0 * 1000))
                self.get_logger().info(f"home done, moving to {target_deg:.2f} deg (pos={target}, time_ms={time_ms})")
                self._servo.move_to(target, time_ms)
                self._state = TurntableStatus.STATE_IDLE
                response.success = True
                response.message = "homed"
            except Exception as exc:
                self._state = TurntableStatus.STATE_ERROR
                response.success = False
                response.message = f"home failed: {exc}"

        elif cmd == TurntableCommand.Request.CMD_MOVE:
            self.get_logger().info("executing MOVE command")
            self._state = TurntableStatus.STATE_MOVING
            try:
                pos = self._servo.deg_to_pos(request.target_deg)
                time_ms = max(200, int(request.duration_s * 1000)) if request.duration_s > 0 else 2000
                self.get_logger().info(f"moving to {request.target_deg:.2f} deg (pos={pos}, time_ms={time_ms})")
                # Restart auto report so the device increments BATCH for this motion stream.
                if self._auto_report_ms > 0:
                    self._servo.start_auto_report(self._auto_report_ms)
                self._servo.move_to(pos, time_ms)
                response.success = True
                response.message = f"moving to {request.target_deg:.1f} deg"
            except Exception as exc:
                self._state = TurntableStatus.STATE_ERROR
                response.success = False
                response.message = f"move failed: {exc}"

        elif cmd == TurntableCommand.Request.CMD_STOP:
            self.get_logger().info("executing STOP command")
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

        self.get_logger().info(f"command result: success={response.success}, message='{response.message}'")
        return response

    def destroy_node(self):
        self._shutdown.set()
        if self._poll_thread is not None:
            try:
                self._poll_thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            self._servo.close()
        except Exception:
            pass
        try:
            super().destroy_node()
        except KeyboardInterrupt:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TurntableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
