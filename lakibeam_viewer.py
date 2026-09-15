#!/usr/bin/env python3
"""LakiBeam LiDAR 网络接收 + Open3D 实时可视化脚本。

连接 LakiBeam 系列（LakiBeam 1/1S/1L）雷达，接收 MSOP UDP 数据包，
解析出距离/角度/回波强度，转换为直角坐标点云并用 Open3D 实时显示。

用法:
    # 0. 网线连接雷达，本机网卡配 192.168.198.1/24（雷达出厂 192.168.198.2）
    #    sudo ip addr add 192.168.198.1/24 dev <网卡名>
    # 1. 浏览器 http://192.168.198.2 确认 laser_enable=True, DataPort=2368
    conda run -n drill_rod_scanner python scripts/lakibeam_viewer.py
    conda run -n drill_rod_scanner python scripts/lakibeam_viewer.py --lidar-ip 192.168.198.2 --port 2368

依赖:
    numpy, open3d（项目 drill_rod_scanner 环境已包含）

协议参考: RichbeamTechnology/Lakibeam_ROS2_Driver (data_type.h / lakibeam1_scan.cpp)
MSOP 包结构 (1248 字节, 小端序):
    42B  UDP/IP 头（本脚本绑定本机端口直接 recv，socket 层已剥离）
    12 × Data Block (100B):
        2B DataFlag (0xEEFF)
        2B Azimuth      (单位 0.01°, uint16)
        16 × MeasuringResult (6B):
            2B Dist_1 + 1B RSSI_1   <- 最强回波
            2B Dist_2 + 1B RSSI_2   <- 最后回波（脚本忽略）
    4B  Timestamp (us)
    2B  Factory
块内 16 点角度线性插值: angle = Azimuth[j] + (Azimuth[j+1]-Azimuth[j])/16 * i
无效数据: DataFlag != 0xEEFF 或 dist == 0 时跳过
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

# ---- MSOP 协议常量 ----
MSOP_DATA_BLOCKS = 12
MSOP_POINTS_PER_BLOCK = 16
MSOP_BLOCK_SIZE = 100  # 2 flag + 2 azimuth + 16*6
# UDP 载荷 = 12×100B Data Block + 4B Timestamp + 2B Factory = 1206 字节。
# （42B 网络头由内核剥离，不进入应用层载荷；tcpdump 实测 length 1206。）
MSOP_PACKET_SIZE = MSOP_DATA_BLOCKS * MSOP_BLOCK_SIZE + 4 + 2
DATA_FLAG = 0xEEFF
INVALID_DIST = 0


@dataclass
class ScanPoint:
    """一圈扫描中的一个有效测距点。"""

    angle: float  # 度 (0.01° 原始值 / 100)
    dist_mm: int
    rssi: int


class MSOPParser:
    """解析 LakiBeam MSOP UDP 数据包。"""

    @staticmethod
    def parse_packet(packet: bytes) -> list[ScanPoint]:
        """解析一个 MSOP 包，返回其中的有效测距点。

        与官方 ROS2 驱动一致：
        - 块内角度线性插值（相邻块 Azimuth 差 / 16）
        - 仅保留 DataFlag == 0xEEFF 且 dist != 0 的点
        """
        if len(packet) < MSOP_PACKET_SIZE:
            return []

        points: list[ScanPoint] = []
        offset = 0  # UDP 载荷直接从 DataFlag 开始，无网络头

        # 解析所有块的 Azimuth，先取前两块角度差计算块内插值分辨率。
        # （与官方 ROS2 驱动一致：resolution = (Azimuth[1] - Azimuth[0]) / 16，
        #  应用于包内所有块，包括第 0 块。）
        azimuths: list[float] = []
        flags: list[int] = []
        for block_idx in range(MSOP_DATA_BLOCKS):
            block_start = offset + block_idx * MSOP_BLOCK_SIZE
            flag = struct.unpack_from("<H", packet, block_start)[0]
            azimuth_raw = struct.unpack_from("<H", packet, block_start + 2)[0]
            flags.append(flag)
            azimuths.append(azimuth_raw / 100.0)

        # 块内角度分辨率：相邻块角度差 / 16（差为正才有效）
        resolution = 0.0
        if len(azimuths) >= 2:
            diff = azimuths[1] - azimuths[0]
            if diff > 0:
                resolution = diff / MSOP_POINTS_PER_BLOCK

        for block_idx in range(MSOP_DATA_BLOCKS):
            flag = flags[block_idx]
            azimuth = azimuths[block_idx]
            block_start = offset + block_idx * MSOP_BLOCK_SIZE

            if flag != DATA_FLAG:
                continue

            for i in range(MSOP_POINTS_PER_BLOCK):
                # 每个测距结果 6 字节：Dist_1(2B) RSSI_1(1B) Dist_2(2B) RSSI_2(1B)
                result_start = block_start + 4 + i * 6
                dist_1, rssi_1 = struct.unpack_from(
                    "<HB", packet, result_start
                )
                if dist_1 == INVALID_DIST:
                    continue  # 无回波

                angle = azimuth + resolution * i
                points.append(ScanPoint(angle=angle, dist_mm=dist_1, rssi=rssi_1))

        return points


def first_block_azimuth(packet: bytes) -> float | None:
    """读取包内第 0 个 Data Block 的方位角（度）；无效包返回 None。"""
    if len(packet) < MSOP_PACKET_SIZE:
        return None
    flag = struct.unpack_from("<H", packet, 0)[0]
    if flag != DATA_FLAG:
        return None
    return struct.unpack_from("<H", packet, 2)[0] / 100.0


def packet_timestamp_us(packet: bytes) -> int | None:
    """读取 MSOP 包尾 4B 微秒时间戳；无效包返回 None。"""
    if len(packet) < MSOP_PACKET_SIZE:
        return None
    flag = struct.unpack_from("<H", packet, 0)[0]
    if flag != DATA_FLAG:
        return None
    timestamp_offset = MSOP_DATA_BLOCKS * MSOP_BLOCK_SIZE
    return struct.unpack_from("<I", packet, timestamp_offset)[0]


def scan_to_xy(points: list[ScanPoint], offset_z_m: float = 0.0) -> np.ndarray:
    """将一圈测距点转换为 (n,3) 直角坐标点云（雷达系）。

    雷达系（安装方式）：x 向下、y 向左、z 向前。
    自转扫描弧在 x-y 平面：0° 指 +x（向下），90° 指 +y（向左）。
    xyz 是雷达系坐标，横装（安装姿态）不改变它；横装变换由
    servo_sweep_scan.to_world（绕世界 y 轴向下转 90°）负责。
    z = offset_z_m 为沿 z 方向的安装偏移（通常 0）。
    距离单位 mm -> m。
    """
    if not points:
        return np.empty((0, 3))
    angles = np.array([p.angle for p in points], dtype=np.float64)
    dists = np.array([p.dist_mm for p in points], dtype=np.float64) / 1000.0
    thetas = np.deg2rad(angles)
    return np.column_stack([
        dists * np.cos(thetas),                        # x 向下（0° 指向 +x）
        dists * np.sin(thetas),                        # y 向左（90° 指向 +y）
        np.full(len(points), offset_z_m, dtype=np.float64),  # z 向前（安装偏移）
    ])


class LakiBeamViewer:
    """接收 LakiBeam UDP 数据流并用 Open3D 实时可视化。"""

    def __init__(
        self,
        host_ip: str = "0.0.0.0",
        port: int = 2368,
        offset_z_m: float = 0.0,
        min_rssi: int = 0,
        max_range_m: float = 50.0,
        debug: bool = False,
    ) -> None:
        self.host_ip = host_ip
        self.port = port
        self.offset_z_m = offset_z_m
        self.min_rssi = min_rssi
        self.max_range_m = max_range_m
        self.debug = debug
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        """绑定本机 UDP 端口，等待雷达数据。"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host_ip, self.port))
        self.sock.settimeout(1.0)  # 1 秒超时，用于主循环退出检查
        print(f"已绑定 UDP {self.host_ip}:{self.port}，等待 LakiBeam 数据...")
        print("提示: 确保雷达已启动测距（Web 配置页 laser_enable=True），且向本机端口发送数据")

    def clear_buffer(self) -> int:
        """清空 UDP 接收缓冲区，返回丢弃的数据包数量。"""
        if self.sock is None:
            return 0
        
        count = 0
        # 临时设置非阻塞模式
        self.sock.settimeout(0.0)
        try:
            while True:
                try:
                    self.sock.recvfrom(65535)
                    count += 1
                except socket.error:
                    break
        finally:
            # 恢复原来的超时设置
            self.sock.settimeout(1.0)
        
        return count

    def receive_scan(self) -> list[ScanPoint] | None:
        """接收数据直到集齐一圈（方位角回绕判定），超时返回 None。

        一圈判定：当前包首块方位角 < 上一包首块方位角，即跨过 360° 边界
        （不依赖从 0° 起扫的假设）。2 秒内无回绕则返回已收集数据防挂起。
        """
        if self.sock is None:
            raise RuntimeError("未连接，请先调用 connect()")

        all_points: list[ScanPoint] = []
        prev_azimuth: float | None = None
        start = time.monotonic()

        while True:
            try:
                packet, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                if all_points:
                    return all_points  # 超时但有数据，返回已收集
                return None

            points = MSOPParser.parse_packet(packet)
            all_points.extend(points)

            az = first_block_azimuth(packet)
            if self.debug:
                print(f"  [debug] 包 {len(packet)}B, 首块方位角 {az}, 解析 {len(points)} 点")
            if az is not None and prev_azimuth is not None and az < prev_azimuth:
                if self.debug:
                    print(f"  [debug] 检测到回绕 {prev_azimuth} -> {az}，一圈完成")
                return all_points
            if az is not None:
                prev_azimuth = az

            if time.monotonic() - start > 2.0:
                if self.debug:
                    print("  [debug] 2 秒未检测到回绕，返回已收集数据")
                return all_points or None

    def visualize(self, fps: int = 20) -> None:
        """Open3D 实时可视化循环，Ctrl+C 或关窗退出。"""
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LakiBeam Live", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        geometry_added = False

        frame_count = 0
        try:
            while vis.poll_events():
                scan = self.receive_scan()
                if scan is None:
                    continue  # 超时无数据，继续等

                pts = scan_to_xy(scan, offset_z_m=self.offset_z_m)
                if pts.shape[0] == 0:
                    continue
                dist = np.linalg.norm(pts[:, :2], axis=1)
                mask = dist <= self.max_range_m
                pts = pts[mask]

                pcd.points = o3d.utility.Vector3dVector(pts)

                if not geometry_added:
                    # 首次有真实数据再添加几何体，否则空包围盒会导致画面空白
                    vis.add_geometry(pcd)
                    ctr = vis.get_view_control()
                    ctr.set_front([0.0, 0.0, 1.0])
                    ctr.set_up([0.0, 1.0, 0.0])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                    geometry_added = True
                else:
                    # 首次 add 时包围盒已基于真实数据建立，扫描范围固定，直接更新即可
                    vis.update_geometry(pcd)
                vis.update_renderer()

                frame_count += 1
                if frame_count % fps == 0:
                    print(f"帧 {frame_count}: {pts.shape[0]} 点")
        except KeyboardInterrupt:
            print("\n用户中断，退出")
        finally:
            vis.destroy_window()
            self.close()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LakiBeam 雷达实时点云可视化")
    parser.add_argument("--lidar-ip", default="192.168.198.2",
                        help="雷达 IP（默认 192.168.198.2）")
    parser.add_argument("--port", type=int, default=2368,
                        help="接收数据端口（默认 2368，需与雷达配置一致）")
    parser.add_argument("--host-ip", default="0.0.0.0",
                        help="绑定本机 IP（默认 0.0.0.0 监听所有网卡）")
    parser.add_argument("--height", type=float, default=0.0,
                        help="雷达安装 z 偏移（米，横向安装相对转盘轴的距离）")
    parser.add_argument("--min-rssi", type=int, default=0,
                        help="回波强度下限过滤（默认 0 不过滤）")
    parser.add_argument("--max-range", type=float, default=50.0,
                        help="最大显示距离（米）")
    parser.add_argument("--fps", type=int, default=20,
                        help="打印帧信息的频率")
    parser.add_argument("--debug", action="store_true",
                        help="打印每个 UDP 包的诊断信息（字节数/方位角/解析点数）")
    args = parser.parse_args()

    print(f"目标雷达: {args.lidar_ip}")
    viewer = LakiBeamViewer(
        host_ip=args.host_ip,
        port=args.port,
        offset_z_m=args.height,
        min_rssi=args.min_rssi,
        max_range_m=args.max_range,
        debug=args.debug,
    )
    viewer.connect()
    viewer.visualize(fps=args.fps)


if __name__ == "__main__":
    main()
