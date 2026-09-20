#!/usr/bin/env python3
"""Open3D 交互微调相机-雷达外参：用外参算出"彩色点云"(与 colorize_pointcloud 同式)，
在 Open3D 窗口里显示；按单键微调外参，立即重算颜色并刷新。

按键（写在终端里, 无需回车）:
    平移(相机系)   w/s = dy -/+      a/d = dx -/+      q/e = dz -/+
    旋转(世界系)   i/k = pitch -/+   j/l = yaw -/+     u/o = roll -/+
    步长           1 = 精细(0.05°/1mm)  2 = 中(0.2°/5mm)  3 = 粗(1°/20mm)
    着色           c = 循环: 图像色 / 冻结色(以零调量的颜色, 最能看出错位) / 深度 / 纯红
    点大小         [ / ]
    帧             n = 下一帧   b = 上一帧
    其它           r = 归零   p = 保存(外参 + 全量彩色云)   x = 退出

用法:
    python3 tune_extrinsics_o3d.py [帧目录...] \
        [--extrinsics config/camera_extrinsics.yaml] [--out /tmp/extrinsics_tuned.yaml]
    不给帧目录时自动选 turntable_output 下"最新且点数最多"的帧。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import queue
import shutil
import sys
import termios
import threading
import tty
import select
import signal

import cv2
import numpy as np
import yaml
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, rotation_z, write_ply_rgb
from verify_colorization import read_ply_xyzrgb

STEPS = [(0.05, 0.001), (0.20, 0.005), (1.00, 0.020)]
MODE_NAMES = ["图像色", "冻结色", "深度", "纯红"]


def rotm(axis: str, deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def delta_matrix(d: dict) -> np.ndarray:
    return rotm("z", d["yaw"]) @ rotm("y", d["pitch"]) @ rotm("x", d["roll"])


def ply_vertex_count(path: str) -> int:
    with open(path) as f:
        for line in f:
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    return 0


def pick_best_frame(root: str = "turntable_output", recent: int = 6) -> str:
    cands = []
    for d in glob.glob(os.path.join(root, "2026*")):
        if not os.path.isdir(d):
            continue
        m, c = os.path.join(d, "merged.ply"), os.path.join(d, "color.png")
        if os.path.exists(m) and os.path.exists(c):
            cands.append((os.path.basename(d), d, ply_vertex_count(m)))
    if not cands:
        raise SystemExit(f"{root} 下没有可用帧")
    cands.sort(key=lambda t: t[0], reverse=True)
    pool = cands[:recent]
    name, path, cnt = max(pool, key=lambda t: t[2])
    print(f"自动选帧: 最近 {len(pool)} 帧中点数最多 -> {name} ({cnt} 点)")
    return path


def colorized(R0, t0, d, photo_angle, pts, img, K, dist):
    """与 colorize_pointcloud.colorize 完全相同的计算（外加微调 d）。返回 (vis, colors uint8)。"""
    t = t0 + np.array([d["dx"], d["dy"], d["dz"]])
    R_full = (R0 @ delta_matrix(d)) @ rotation_z(photo_angle)
    proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(R_full)[0], t, K, dist)
    proj = proj.reshape(-1, 2)
    cam_z = (R_full @ pts.T + t.reshape(3, 1)).T[:, 2]
    h, w = img.shape[:2]
    finite = np.isfinite(proj) & (np.abs(proj) < 1.0e6)
    good = finite[:, 0] & finite[:, 1]
    safe = np.where(good[:, None], proj, 0.0)
    u = np.round(safe[:, 0]).astype(int)
    v = np.round(safe[:, 1]).astype(int)
    vis = good & (cam_z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    cols = np.full((len(pts), 3), 30, np.uint8)
    cols[vis] = img[v[vis], u[vis]][:, ::-1]
    return vis, cols


def key_reader(q: "queue.Queue[str]", stop: threading.Event) -> None:
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception as exc:
        print(f"⚠️  终端不是 tty（{exc}）→ 无法读按键；请在交互式终端里运行 ✗")
        return
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                q.put(sys.stdin.read(1))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def install_signal_handlers(stop: threading.Event) -> None:
    """Ctrl+C / SIGTERM 时干净退出（渲染循环会占住 GIL, 默认信号处理常被挂住）。"""
    def handler(signum, _frame):
        print(f"\n收到信号 {signum} → 正在干净退出…（退出后如终端异常, 执行 stty sane）", flush=True)
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


class Tuner:
    def __init__(self, dirs, ex_path, info_path, photo_angle, out_path, max_pts):
        self.dirs, self.i = dirs, 0
        self.photo_angle, self.out_path, self.max_pts = photo_angle, out_path, max_pts
        self.K, self.dist = load_camera_info(info_path)
        ex = yaml.safe_load(open(ex_path))["lidar_to_camera"]
        self.R0 = np.array(ex["rotation_matrix"], float).reshape(3, 3)
        self.t0 = np.array(ex["translation"], float)
        self.d = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, dz=0.0)
        self.step_i, self.mode, self.size = 1, 0, 2.0
        print(f"基准外参 {ex_path}   t = {np.round(self.t0, 5).tolist()}")
        self._load_frame()
        self.frozen = self._base_cols.copy()

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.pts)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = self.size

        print(f"DISPLAY={os.environ.get('DISPLAY')!r}  XAUTHORITY={os.environ.get('XAUTHORITY')!r}")
        if not os.environ.get("DISPLAY"):
            print("⚠️  DISPLAY 为空 → Open3D 会退回离屏渲染, 窗口不会出现 ✗\n"
                  "   请在能看到图像的 shell 里运行（容器内可用: export DISPLAY=:0; "
                  "export XAUTHORITY=/root/.Xauthority 或宿主机用户家目录下的 .Xauthority）")
        self.app = gui.Application.instance
        self.app.initialize()
        self.win = o3d.visualization.O3DVisualizer("tune extrinsics (终端按单键)", 1280, 800)
        self.win.add_geometry("cloud", self.pcd, self.mat)
        self.win.reset_camera_to_default()
        self.app.add_window(self.win)
        self.apply_colors(verbose=False)

        self.q: "queue.Queue[str]" = queue.Queue()
        self.stop = threading.Event()
        install_signal_handlers(self.stop)
        self.th = threading.Thread(target=key_reader, args=(self.q, self.stop), daemon=True)
        self.th.start()

    def _load_frame(self):
        d = self.dirs[self.i]
        self.img = cv2.imread(os.path.join(d, "color.png"))
        if self.img is None:
            raise SystemExit(f"读不到 {d}/color.png")
        xyz, _ = read_ply_xyzrgb(os.path.join(d, "merged.ply"))
        if len(xyz) > self.max_pts:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), self.max_pts, replace=False)]
        self.pts = xyz
        self._base_cols = np.full((len(xyz), 3), 30, np.uint8)
        R_full = self.R0 @ rotation_z(self.photo_angle)
        proj, _ = cv2.projectPoints(xyz, cv2.Rodrigues(R_full)[0], self.t0, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        cz = (R_full @ xyz.T + self.t0.reshape(3, 1)).T[:, 2]
        h, w = self.img.shape[:2]
        u = np.clip(np.round(proj[:, 0]).astype(int), -1, w)
        v = np.clip(np.round(proj[:, 1]).astype(int), -1, h)
        vis = (cz > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(proj).all(1)
        self._base_cols[vis] = self.img[v[vis], u[vis]][:, ::-1]
        if hasattr(self, "pcd"):
            self.pcd.points = o3d.utility.Vector3dVector(self.pts)
            self.apply_colors(verbose=False)
        print(f"帧 {self.i+1}/{len(self.dirs)}: {os.path.basename(d)}  {len(self.pts)} 点  "
              f"(基准上色 {int(vis.sum())} 点)")

    def colors_now(self) -> np.ndarray:
        if self.mode == 0:
            return colorized(self.R0, self.t0, self.d, self.photo_angle,
                             self.pts, self.img, self.K, self.dist)[1]
        if self.mode == 1:
            return self.frozen
        if self.mode == 2:
            _, z = colorized(self.R0, self.t0, self.d, self.photo_angle,
                             self.pts, self.img, self.K, self.dist)
            R_full = (self.R0 @ delta_matrix(self.d)) @ rotation_z(self.photo_angle)
            z = (R_full @ self.pts.T + self.t0.reshape(3, 1)).T[:, 2]
            zn = np.clip((z - 0.5) / 3.5, 0, 1)
            import matplotlib.pyplot as plt
            return (plt.get_cmap("turbo")(zn)[:, :3] * 255).astype(np.uint8)
        return np.tile(np.array([[235, 60, 60]], np.uint8), (len(self.pts), 1))

    def apply_colors(self, verbose: bool = True):
        cols = self.colors_now().astype(float) / 255.0
        self.pcd.colors = o3d.utility.Vector3dVector(cols)
        try:
            self.win.add_geometry("cloud", self.pcd, self.mat, reset_bounding_box=False)
        except TypeError:
            self.win.add_geometry("cloud", self.pcd, self.mat)
        d = self.d
        try:
            self.win.title = (f"yaw={d['yaw']:+.2f} pitch={d['pitch']:+.2f} roll={d['roll']:+.2f} | "
                              f"dx={d['dx']*1000:+.0f} dy={d['dy']*1000:+.0f} dz={d['dz']*1000:+.0f}mm | "
                              f"{MODE_NAMES[self.mode]}")
        except Exception:
            pass
        if verbose:
            d = self.d
            vis, _ = colorized(self.R0, self.t0, d, self.photo_angle,
                               self.pts, self.img, self.K, self.dist)
            print(f"[{os.path.basename(self.dirs[self.i])}] yaw={d['yaw']:+.3f} pitch={d['pitch']:+.3f} "
                  f"roll={d['roll']:+.3f} | dx={d['dx']*1000:+.0f} dy={d['dy']*1000:+.0f} "
                  f"dz={d['dz']*1000:+.0f}mm | step{self.step_i+1} {MODE_NAMES[self.mode]} "
                  f"dot={self.size:g} | colored={int(vis.sum())}", flush=True)

    def save(self):
        R = self.R0 @ delta_matrix(self.d)
        t = self.t0 + np.array([self.d["dx"], self.d["dy"], self.d["dz"]])
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                            "translation": t.tolist()},
                        "note": "tuned by tune_extrinsics_o3d.py"},
                       open(self.out_path, "w"), sort_keys=False)
        full_pts, _ = read_ply_xyzrgb(os.path.join(self.dirs[self.i], "merged.ply"))
        keep, self.pts = self.pts, full_pts
        try:
            _, cols = colorized(self.R0, self.t0, self.d, self.photo_angle,
                                full_pts, self.img, self.K, self.dist)
        finally:
            self.pts = keep
        col_out = os.path.join(os.path.dirname(self.out_path) or ".", "colored_tuned.ply")
        write_ply_rgb(col_out, full_pts, cols)
        print("\n=== 微调结果（可直接报给助手）===")
        print(f"  旋转: yaw={self.d['yaw']:+.3f}°  pitch={self.d['pitch']:+.3f}°  roll={self.d['roll']:+.3f}°")
        print(f"  平移: dx={self.d['dx']*1000:+.1f}mm  dy={self.d['dy']*1000:+.1f}mm  dz={self.d['dz']*1000:+.1f}mm")
        print(f"  t = {np.round(t, 5).tolist()}")
        print(f"  外参: {self.out_path}")
        print(f"  彩色云(全量): {col_out}  上色 {int((cols != 30).any(1).sum())} 点")

    def run(self):
        print("按键: w/s a/d q/e 平移 | i/k j/l u/o 旋转 | 1/2/3 步长 | c 着色 | [ ] 点大小 | "
              "n/b 换帧 | r 归零 | p 保存 | x 退出   (窗口里按 Q 也能关)")
        while not self.stop.is_set():
            while not self.q.empty():
                k = self.q.get()
                deg, mm = STEPS[self.step_i]
                changed = True
                if k == "w": self.d["dy"] -= mm
                elif k == "s": self.d["dy"] += mm
                elif k == "a": self.d["dx"] -= mm
                elif k == "d": self.d["dx"] += mm
                elif k == "q": self.d["dz"] -= mm
                elif k == "e": self.d["dz"] += mm
                elif k == "i": self.d["pitch"] -= deg
                elif k == "k": self.d["pitch"] += deg
                elif k == "j": self.d["yaw"] -= deg
                elif k == "l": self.d["yaw"] += deg
                elif k == "u": self.d["roll"] -= deg
                elif k == "o": self.d["roll"] += deg
                elif k in ("1", "2", "3"): self.step_i = int(k) - 1
                elif k == "c": self.mode = (self.mode + 1) % 4
                elif k == "]": self.size = min(12.0, self.size + 2.0)
                elif k == "[": self.size = max(1.0, self.size - 2.0)
                elif k == "r":
                    for kk in self.d: self.d[kk] = 0.0
                elif k == "p": self.save()
                elif k == "n": self.i = (self.i + 1) % len(self.dirs); self._load_frame(); self.frozen = self._base_cols.copy()
                elif k == "b": self.i = (self.i - 1) % len(self.dirs); self._load_frame(); self.frozen = self._base_cols.copy()
                elif k in ("x", "\x1b", "\x03"): self.quit(); return
                else: changed = False
                if k in ("]", "["):
                    self.mat.point_size = self.size
                    self.apply_colors(verbose=False)
                if changed:
                    self.apply_colors()
            self.app.run_one_tick()
        self.quit()

    def quit(self):
        self.stop.set()
        try:
            self.app.quit()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Open3D 交互微调外参（按键即重算彩色点云）")
    ap.add_argument("pose_dirs", nargs="*")
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--camera-info", default="config/camera_info.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--out", default="/tmp/extrinsics_tuned.yaml")
    ap.add_argument("--max-points", type=int, default=300000)
    a = ap.parse_args()
    dirs = a.pose_dirs if a.pose_dirs else [pick_best_frame()]
    Tuner(dirs, a.extrinsics, a.camera_info, a.photo_angle, a.out, a.max_points).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
