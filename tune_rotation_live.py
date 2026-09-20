#!/usr/bin/env python3
"""键盘微调 yaw/pitch/roll + tx/ty/tz 并实时重渲染最后一帧 colored.ply 供目测。

键位 (旋转每次 0.2°, 平移每次 1mm, 均叠加在合并系/雷达侧, 右手系):
  a / d   yaw   负/正   (绕世界Z, 俯视逆时针为正)
  s / w   pitch 负/正   (绕世界X)
  q / e   roll  负/正   (绕世界Y)
  f / h   tx    负/正   (mm)
  t / g   ty    正/负   (mm, t=+ g=-)
  v / b   tz    负/正   (mm)
  r       清零全部角度与平移
  x       退出 (微调不写入正式外参)

用法(需交互式终端):
  docker exec -it silly_bell bash -lc 'cd /workspace && python3 tune_rotation_live.py'
  (容器名以 docker ps 实际为准; 数据帧目录默认取 turntable_output/20260918_* 最后一帧)
"""

import contextlib
import glob
import io
import os
import sys
import termios
import tty

import numpy as np
import yaml

BASE_EXTRINSICS = "config/camera_extrinsics.yaml"
CAMERA_INFO = "config/camera_info.yaml"
TMP_EXTRINSICS = "/tmp/ext_rotation_tune.yaml"
STEP_DEG = 0.2
STEP_MM = 1.0


def rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> int:
    dirs = sorted(glob.glob("turntable_output/20260918_*"))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "merged.ply"))]
    if not dirs:
        print("未找到数据帧")
        return 1
    pose = dirs[-1]
    ex = yaml.safe_load(open(BASE_EXTRINSICS))["lidar_to_camera"]
    R0 = np.array(ex["rotation_matrix"]).reshape(3, 3)
    t = np.array(ex["translation"])

    from colorize_pointcloud import colorize

    yaw = pitch = roll = 0.0
    dx = dy = dz = 0.0
    print(f"目标帧: {pose}  (基准: {BASE_EXTRINSICS}, 旋转每次 {STEP_DEG}°, 平移每次 {STEP_MM}mm)")
    print("键位: a/d=yaw  s/w=pitch  q/e=roll  f/h=tx  t/g=ty  v/b=tz  r=清零  x=退出\n")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    try:
        while True:
            ch = sys.stdin.read(1)
            if ch == "x":
                break
            if ch == "r":
                yaw = pitch = roll = 0.0
                dx = dy = dz = 0.0
            elif ch == "a":
                yaw -= STEP_DEG
            elif ch == "d":
                yaw += STEP_DEG
            elif ch == "s":
                pitch -= STEP_DEG
            elif ch == "w":
                pitch += STEP_DEG
            elif ch == "q":
                roll -= STEP_DEG
            elif ch == "e":
                roll += STEP_DEG
            elif ch == "f":
                dx -= STEP_MM
            elif ch == "h":
                dx += STEP_MM
            elif ch == "t":
                dy += STEP_MM
            elif ch == "g":
                dy -= STEP_MM
            elif ch == "v":
                dz -= STEP_MM
            elif ch == "b":
                dz += STEP_MM
            else:
                continue
            R = R0 @ rot_z(yaw) @ rot_x(pitch) @ rot_y(roll)
            t_tune = t + np.array([dx, dy, dz]) / 1000.0
            yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                                "translation": t_tune.tolist()}},
                           open(TMP_EXTRINSICS, "w"), sort_keys=False)
            with contextlib.redirect_stdout(io.StringIO()):
                colorize(pose, TMP_EXTRINSICS, CAMERA_INFO, 90.0)
            sys.stdout.write(f"\r yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f} | "
                             f"tx={dx:+.0f} ty={dy:+.0f} tz={dz:+.0f}mm"
                             f" -> {os.path.join(pose, 'colored.ply')}    ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(f"\n退出 ✓ 最终: yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f} | "
          f"tx={dx:+.0f} ty={dy:+.0f} tz={dz:+.0f}mm (未写入正式外参, 满意后报数给我)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
