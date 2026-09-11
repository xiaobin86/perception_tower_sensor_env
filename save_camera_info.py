#!/usr/bin/env python3
"""保存相机内参到配置文件。

用法：
    source install/setup.bash
    python3 save_camera_info.py

会订阅 /camera/color/camera_info，保存到 config/camera_info.yaml
"""

from __future__ import annotations

import os
import sys
import argparse
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class CameraInfoSaver(Node):
    def __init__(self, output_path: str):
        super().__init__("camera_info_saver")
        self._output_path = output_path
        self._sub = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._on_camera_info, 10
        )
        self._received = False
        self.get_logger().info("等待 /camera/color/camera_info ...")

    def _on_camera_info(self, msg: CameraInfo):
        if self._received:
            return
        self._received = True

        data = {
            "header": {
                "frame_id": msg.header.frame_id,
            },
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": msg.distortion_model,
            "d": [float(x) for x in msg.d],
            "k": [float(x) for x in msg.k],
            "r": [float(x) for x in msg.r],
            "p": [float(x) for x in msg.p],
            "binning_x": int(msg.binning_x),
            "binning_y": int(msg.binning_y),
            "roi": {
                "x_offset": int(msg.roi.x_offset),
                "y_offset": int(msg.roi.y_offset),
                "height": int(msg.roi.height),
                "width": int(msg.roi.width),
                "do_rectify": bool(msg.roi.do_rectify),
            },
        }

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(f"已保存相机内参: {self._output_path}")
        self.get_logger().info(f"分辨率: {msg.width}x{msg.height}")
        self.get_logger().info(f"K: {list(msg.k)}")
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="保存相机内参")
    parser.add_argument("data_dir", nargs='?', default=".", help="数据目录（默认当前目录）")
    args = parser.parse_args()

    output_path = os.path.join(args.data_dir, "config", "camera_info.yaml")

    rclpy.init()
    node = CameraInfoSaver(output_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
