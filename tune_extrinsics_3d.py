#!/usr/bin/env python3
"""微调相机-雷达外参（纯 CPU 渲染 · 不会碰 GPU）：3D 视图里显示"彩色点云"，
按键微调外参后立即重算颜色并刷新。

为什么不用 Open3D: 你的会话是 Wayland+XWayland, Open3D 走 Vulkan/GL, 一开窗容易把
合成器/VS Code 挂死 ✗。本工具用 matplotlib 的 CPU 渲染 + 窗口内按键（不接管终端）✓。

投影与着色公式与 colorize_pointcloud.colorize 完全一致（已逐点验证）。

按键（先在窗口里点一下让它获得焦点）:
   平移(相机系)  w/s = dy -/+     a/d = dx -/+     q/e = dz -/+
   旋转(世界系)  i/k = pitch -/+  j/l = yaw -/+    u/o = roll -/+
   步长          1 = 精细(0.05°/1mm)  2 = 中(0.2°/5mm)  3 = 粗(1°/20mm)
   着色          c = 循环: 图像色 / 冻结色(最能看出错位) / 纯红
   其它          r = 归零   p = 保存   n/b = 下一帧/上一帧   Esc = 退出

用法:  python3 tune_extrinsics_3d.py [帧目录...] [--extrinsics ...] [--out /tmp/extrinsics_tuned.yaml]
       不给目录时自动选 turntable_output 下"最新且点数最多"的帧。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, rotation_z, write_ply_rgb
from verify_colorization import read_ply_xyzrgb

STEPS = [(0.05, 0.001), (0.20, 0.005), (1.00, 0.020)]
MODE_NAMES = ["image", "frozen", "red"]


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
        if os.path.exists(os.path.join(d, "merged.ply")) and os.path.exists(os.path.join(d, "color.png")):
            cands.append((os.path.basename(d), d, ply_vertex_count(os.path.join(d, "merged.ply"))))
    if not cands:
        raise SystemExit(f"{root} 下没有可用帧")
    cands.sort(key=lambda t: t[0], reverse=True)
    pool = cands[:recent]
    name, path, cnt = max(pool, key=lambda t: t[2])
    print(f"自动选帧: 最近 {len(pool)} 帧中点数最多 -> {name} ({cnt} 点)")
    return path


def colorize_like(R0, t0, d, photo_angle, pts, img, K, dist):
    """与 colorize_pointcloud.colorize 相同的计算，外加微调 d。返回 (vis, colors uint8)。"""
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


class Tuner:
    def __init__(self, dirs, ex_path, info_path, photo_angle, out_path, max_pts):
        self.dirs, self.i = dirs, 0
        self.photo_angle, self.out_path = photo_angle, out_path
        self.max_pts = max_pts
        self.K, self.dist = load_camera_info(info_path)
        ex = yaml.safe_load(open(ex_path))["lidar_to_camera"]
        self.R0 = np.array(ex["rotation_matrix"], float).reshape(3, 3)
        self.t0 = np.array(ex["translation"], float)
        self.d = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, dz=0.0)
        self.step_i, self.mode, self.dotsize = 1, 0, 2.0
        print(f"基准外参 {ex_path}   t = {np.round(self.t0, 5).tolist()}")
        self._load_frame()
        self.frozen = self._base_cols.copy()

        self.view = dict(yaw=-2.1, pitch=0.30, dist=6.0, focal=1.15)
        self.target = self.pts.mean(0) if len(self.pts) else np.zeros(3)
        self.drag = None
        self.fig = plt.figure(figsize=(13, 9), facecolor="black")
        self.ax = self.fig.add_axes([0, 0, 1, 0.94])
        self.ax.set_axis_off()                      # 不要任何坐标框/刻度
        self.im = self.ax.imshow(np.zeros((10, 10, 3), np.uint8), aspect="auto")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.apply_colors(first=True)
        self._redraw()

    def _load_frame(self):
        d = self.dirs[self.i]
        self.img = cv2.imread(os.path.join(d, "color.png"))
        if self.img is None:
            raise SystemExit(f"读不到 {d}/color.png")
        xyz, _ = read_ply_xyzrgb(os.path.join(d, "merged.ply"))
        if self.max_pts and len(xyz) > self.max_pts:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), self.max_pts, replace=False)]
        self.pts = xyz
        _, self._base_cols = colorize_like(self.R0, self.t0, self.d, self.photo_angle,
                                           self.pts, self.img, self.K, self.dist)
        print(f"帧 {self.i+1}/{len(self.dirs)}: {os.path.basename(d)}  {len(self.pts)} 点")

    def colors_now(self):
        if self.mode == 0:
            return colorize_like(self.R0, self.t0, self.d, self.photo_angle,
                                 self.pts, self.img, self.K, self.dist)[1]
        if self.mode == 1:
            return self.frozen
        return np.tile(np.array([[235, 60, 60]], np.uint8), (len(self.pts), 1))

    def apply_colors(self, first: bool = False):
        self.cols = self.colors_now().astype(np.uint8)

    def _camera(self):
        v = self.view
        cp, sp = np.cos(v["pitch"]), np.sin(v["pitch"])
        eye = self.target + v["dist"] * np.array([cp * np.cos(v["yaw"]), cp * np.sin(v["yaw"]), sp])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd) + 1e-12
        up0 = np.array([0.0, 0.0, 1.0])
        if abs(float(fwd @ up0)) > 0.99:
            up0 = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up0); right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def render(self):
        H, W = 840, 1280
        canvas = np.zeros((H, W, 3), np.uint8)
        pts = self.pts
        if len(pts) == 0:
            return canvas
        eye, fwd, right, up = self._camera()
        rel = pts - eye
        zc = rel @ fwd
        front = zc > 1e-3
        f = self.view["focal"] * min(W, H)
        u = np.full(len(pts), -1.0)
        vv = np.full(len(pts), -1.0)
        xc = rel[front] @ right
        yc = rel[front] @ up
        u[front] = W / 2 + f * xc / zc[front]
        vv[front] = H / 2 - f * yc / zc[front]
        ui = np.round(u).astype(int)
        vi = np.round(vv).astype(int)
        ok = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return canvas
        order = idx[np.argsort(-zc[idx])]          # 远的先画, 近的覆盖
        canvas[vi[order], ui[order]] = self.cols[order]
        if self.dotsize > 1:
            from scipy import ndimage
            k = int(self.dotsize) - 1
            mask = np.zeros((H, W), bool)
            mask[vi[order], ui[order]] = True
            mask = ndimage.binary_dilation(mask, iterations=k)
            canvas = ndimage.grey_dilation(canvas, size=(2 * k + 1, 2 * k + 1, 1))
            canvas[~mask] = 0
        return canvas

    def _redraw(self):
        d = self.d
        self.im.set_data(self.render())
        self.im.set_extent((0, 1280, 840, 0))
        self.ax.set_title(
            f"frame {self.i+1}/{len(self.dirs)}  {os.path.basename(self.dirs[self.i])}"
            f"    step[{self.step_i+1}]={STEPS[self.step_i][0]}deg/{STEPS[self.step_i][1]*1000:.0f}mm"
            f"    color={MODE_NAMES[self.mode]}    dots={self.dotsize:.0f}    {len(self.pts)} pts\n"
            f"yaw={d['yaw']:+.3f}  pitch={d['pitch']:+.3f}  roll={d['roll']:+.3f} deg    "
            f"dx={d['dx']*1000:+.1f}  dy={d['dy']*1000:+.1f}  dz={d['dz']*1000:+.1f} mm    "
            f"t={np.round(self.t0 + np.array([d['dx'], d['dy'], d['dz']]), 5).tolist()}",
            family="monospace", fontsize=9.5, color="white")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _zoom(self, factor: float):
        self.view["dist"] = float(np.clip(self.view["dist"] * factor, 0.5, 80.0))
        self._redraw()

    def _rotate(self, dyaw: float = 0.0, dpitch: float = 0.0):
        self.view["yaw"] += dyaw
        self.view["pitch"] = float(np.clip(self.view["pitch"] + dpitch, -1.45, 1.45))
        self._redraw()

    def on_scroll(self, ev):
        if ev.inaxes is self.ax:
            self._zoom(0.88 if ev.button == "up" else 1.14)

    def on_press(self, ev):
        if ev.inaxes is self.ax:
            self.drag = (ev.x, ev.y, ev.key == "shift")

    def on_motion(self, ev):
        if self.drag is None or ev.x is None:
            return
        dx, dy, pan = self.drag
        mx, my = ev.x - dx, ev.y - dy
        self.drag = (ev.x, ev.y, pan)
        if pan:
            _, _, right, up = self._camera()
            s = self.view["dist"] * 0.0022
            self.target = self.target - (mx * right + my * up) * s
        else:
            self.view["yaw"] += mx * 0.008
            self.view["pitch"] = float(np.clip(self.view["pitch"] - my * 0.008, -1.45, 1.45))
        self._redraw()

    def on_release(self, ev):
        self.drag = None

    def on_key(self, ev):
        k = ev.key
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
        elif k == "c": self.mode = (self.mode + 1) % 3
        elif k == "]": self.dotsize = min(6.0, self.dotsize + 1)
        elif k == "[": self.dotsize = max(1.0, self.dotsize - 1)
        elif k == "r":
            for kk in self.d: self.d[kk] = 0.0
        elif k == "p": self.save()
        elif k == "n": self.i = (self.i + 1) % len(self.dirs); self._load_frame(); self.frozen = self._base_cols.copy()
        elif k == "b": self.i = (self.i - 1) % len(self.dirs); self._load_frame(); self.frozen = self._base_cols.copy()
        elif k in ("+", "="): self._zoom(0.85); changed = False
        elif k == "-": self._zoom(1.18); changed = False
        elif k == "left": self._rotate(dyaw=-0.15); changed = False
        elif k == "right": self._rotate(dyaw=+0.15); changed = False
        elif k == "up": self._rotate(dpitch=+0.1); changed = False
        elif k == "down": self._rotate(dpitch=-0.1); changed = False
        elif k == "home":
            self.view.update(yaw=-2.1, pitch=0.30, dist=6.0, focal=1.15)
            self.target = self.pts.mean(0); self._redraw(); changed = False
        elif k == "escape": plt.close(self.fig); return
        elif k in (" ", "enter"): pass
        else: changed = False
        if changed:
            if k in ("n", "b"):
                self.sc.remove()
                self.apply_colors(first=True)
            else:
                self.apply_colors()
            d = self.d
            print(f"[{os.path.basename(self.dirs[self.i])}] yaw={d['yaw']:+.3f} pitch={d['pitch']:+.3f} "
                  f"roll={d['roll']:+.3f} | dx={d['dx']*1000:+.0f} dy={d['dy']*1000:+.0f} "
                  f"dz={d['dz']*1000:+.0f}mm | step{self.step_i+1} {MODE_NAMES[self.mode]}", flush=True)
            self._redraw()

    def save(self):
        R = self.R0 @ delta_matrix(self.d)
        t = self.t0 + np.array([self.d["dx"], self.d["dy"], self.d["dz"]])
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                            "translation": t.tolist()},
                        "note": "tuned by tune_extrinsics_3d.py"},
                       open(self.out_path, "w"), sort_keys=False)
        full_pts, _ = read_ply_xyzrgb(os.path.join(self.dirs[self.i], "merged.ply"))
        _, cols = colorize_like(self.R0, self.t0, self.d, self.photo_angle,
                                full_pts, self.img, self.K, self.dist)
        col_out = os.path.join(os.path.dirname(self.out_path) or ".", "colored_tuned.ply")
        write_ply_rgb(col_out, full_pts, cols)
        print("\n=== 微调结果（可直接报给助手）===")
        print(f"  旋转: yaw={self.d['yaw']:+.3f}°  pitch={self.d['pitch']:+.3f}°  roll={self.d['roll']:+.3f}°")
        print(f"  平移: dx={self.d['dx']*1000:+.1f}mm  dy={self.d['dy']*1000:+.1f}mm  dz={self.d['dz']*1000:+.1f}mm")
        print(f"  t = {np.round(t, 5).tolist()}")
        print(f"  外参: {self.out_path}")
        print(f"  彩色云(全量): {col_out}  上色 {int((cols != 30).any(1).sum())} 点")

    def run(self):
        print("按键(先点一下窗口): w/s a/d q/e 平移 | i/k j/l u/o 旋转 | 1/2/3 步长 | c 着色 | "
              "n/b 换帧 | r 归零 | p 保存 | Esc 退出\n"
              "视图: 左键拖拽=旋转 | Shift+拖拽=平移 | 滚轮=缩放 | 方向键=旋转 | +/-=缩放 | "
              "[ ]=点大小 | Home=复位视图")
        plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(description="3D 微调外参（纯 CPU, 不碰 GPU）")
    ap.add_argument("pose_dirs", nargs="*")
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--camera-info", default="config/camera_info.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--out", default="/tmp/extrinsics_tuned.yaml")
    ap.add_argument("--max-points", type=int, default=0, help="0 = 全部点")
    a = ap.parse_args()
    dirs = a.pose_dirs if a.pose_dirs else [pick_best_frame()]
    Tuner(dirs, a.extrinsics, a.camera_info, a.photo_angle, a.out, a.max_points).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
