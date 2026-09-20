#!/usr/bin/env python3
"""键盘微调相机-雷达外参：点云投到照片上，实时显示调量，满意即保存。

渲染: 同一张画布上叠照片 + 雷达点云投影（可按深度或按照片采样色着色）
投影公式与 colorize_pointcloud 一致:
    p_cam = R @ (dR_world @ Rz(photo) @ p_merged) + t + dt
    dR_world = Rz(yaw) @ Ry(pitch) @ Rx(roll)     （绕世界 X/Y/Z）

按键（每按一次 = 一个步长）:
    平移(相机系)   w/s = dy -/+      a/d = dx -/+      q/e = dz -/+
    旋转(世界系)   i/k = pitch -/+   j/l = yaw -/+     u/o = roll -/+
    步长           1 = 精细(0.05°/1mm)   2 = 中(0.2°/5mm)   3 = 粗(1°/20mm)
    其它           c = 切换着色    ←/→ = 换帧    r = 归零    p = 保存    Esc = 退出

用法:
    python3 tune_extrinsics_gui.py turntable_output/20260918_062733 \
        [其它帧...] [--extrinsics config/camera_extrinsics.yaml] [--apply]
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
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import load_camera_info, rotation_z, write_ply_rgb
from verify_colorization import read_ply_xyzrgb

STEPS = [(0.05, 0.001), (0.20, 0.005), (1.00, 0.020)]


def rotm(axis: str, deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def ply_vertex_count(path: str) -> int:
    with open(path) as f:
        for line in f:
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    return 0


def pick_best_frame(root: str = "turntable_output", recent: int = 6) -> str:
    """在最近的几帧里挑点数最多的那帧（"最新 + 点数最多"）。"""
    cands = []
    for d in glob.glob(os.path.join(root, "2026*")):
        if not os.path.isdir(d):
            continue
        m, c = os.path.join(d, "merged.ply"), os.path.join(d, "color.png")
        if os.path.exists(m) and os.path.exists(c):
            cands.append((os.path.basename(d), d, ply_vertex_count(m)))
    if not cands:
        raise SystemExit(f"{root} 下没有可用的帧（需 merged.ply + color.png）")
    cands.sort(key=lambda t: t[0], reverse=True)
    pool = cands[:recent]
    name, path, cnt = max(pool, key=lambda t: t[2])
    print(f"自动选帧：最近 {len(pool)} 帧中点数最多 → {name} ({cnt} 点)")
    for nm, _, k in pool:
        print(f"    {nm}: {k} 点" + ("   ← 选中" if nm == name else ""))
    return path


def delta_matrix(d: dict) -> np.ndarray:
    return rotm("z", d["yaw"]) @ rotm("y", d["pitch"]) @ rotm("x", d["roll"])


def project_points(R0, t0, d, photo_angle, pts, K, dist):
    """与 colorize_pointcloud 相同的投影链，外加微调 d。"""
    P = (R0 @ (delta_matrix(d) @ ((pts @ rotm("z", photo_angle).T).T))).T \
        + t0 + np.array([d["dx"], d["dy"], d["dz"]])
    uv, _ = cv2.projectPoints(P, np.zeros(3), t0 + np.array([d["dx"], d["dy"], d["dz"]]), K, dist)
    uv = uv.reshape(-1, 2)
    z = P[:, 2]
    ok = (z > 0.3) & (z < 8.0)
    return uv[ok], z[ok]


class Tuner:
    def __init__(self, dirs, ex_path, info_path, photo_angle, out_path, max_pts, apply_):
        self.dirs, self.i = dirs, 0
        self.photo_angle, self.out_path, self.max_pts, self.apply = photo_angle, out_path, max_pts, apply_
        self.K, self.dist = load_camera_info(info_path)
        ex = yaml.safe_load(open(ex_path))["lidar_to_camera"]
        self.R0 = np.array(ex["rotation_matrix"], float).reshape(3, 3)
        self.t0 = np.array(ex["translation"], float)
        self.d = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, dz=0.0)
        self.step_i, self.mode, self.size = 1, 2, 0
        print(f"基准外参 {ex_path}\n  t = {np.round(self.t0, 5).tolist()}")
        self._load_frame()

        self.fig = plt.figure(figsize=(14, 9))
        self.ax = self.fig.add_axes([0.02, 0.26, 0.96, 0.70])
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.composite = self.img[:, :, ::-1].copy()
        self.im = self.ax.imshow(self.composite)
        self.hud = self.fig.text(0.02, 0.20, "", family="monospace", fontsize=11,
                                 va="top", linespacing=1.6)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()

    def _load_frame(self):
        d = self.dirs[self.i]
        self.img = cv2.imread(os.path.join(d, "color.png"))
        if self.img is None:
            raise SystemExit(f"读不到 {d}/color.png")
        xyz, _ = read_ply_xyzrgb(os.path.join(d, "merged.ply"))
        if len(xyz) > self.max_pts:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), self.max_pts, replace=False)]
        self.pts = xyz
        print(f"帧 {self.i+1}/{len(self.dirs)}: {os.path.basename(d)}  {len(self.pts)} 点")

    def colorize_like(self):
        """与 colorize_pointcloud.colorize 完全相同的计算（外加微调 d）。"""
        t = self.t0 + np.array([self.d["dx"], self.d["dy"], self.d["dz"]])
        R_full = (self.R0 @ delta_matrix(self.d)) @ rotation_z(self.photo_angle)
        rvec = cv2.Rodrigues(R_full)[0]
        proj, _ = cv2.projectPoints(self.pts, rvec, t, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        cam_z = (R_full @ self.pts.T + t.reshape(3, 1)).T[:, 2]
        h, w = self.img.shape[:2]
        finite = np.isfinite(proj) & (np.abs(proj) < 1.0e6)
        good = finite[:, 0] & finite[:, 1]
        safe = np.where(good[:, None], proj, 0.0)
        u = np.round(safe[:, 0]).astype(int)
        v = np.round(safe[:, 1]).astype(int)
        vis = good & (cam_z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cols = np.full((len(self.pts), 3), 30, np.uint8)
        cols[vis] = self.img[v[vis], u[vis]][:, ::-1]
        return u, v, vis, cols

    def write_colored(self, full: bool = False):
        """写出彩色云。full=True 时读全量 merged.ply（保存时用），否则用当前子采样（每键刷新用）。"""
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        out = os.path.join(os.path.dirname(self.out_path) or ".", "colored_tuned.ply")
        if full:
            pts, _ = read_ply_xyzrgb(os.path.join(self.dirs[self.i], "merged.ply"))
            keep = self.pts
            self.pts = pts
            try:
                _, _, _, cols = self.colorize_like()
            finally:
                self.pts = keep
            write_ply_rgb(out, pts, cols)
            return out, int((cols != 30).any(axis=1).sum())
        _, _, _, cols = self.colorize_like()
        write_ply_rgb(out, self.pts, cols)
        return out, int((cols != 30).any(axis=1).sum())

    def update(self):
        u, v, vis, cols = self.colorize_like()
        H, W = self.img.shape[:2]
        hit = np.zeros((H, W), bool)
        hit[v[vis], u[vis]] = True
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[v[vis], u[vis]] = cols[vis]
        if self.size > 0:
            k = 2 * self.size + 1
            hit = ndimage.binary_dilation(hit, iterations=self.size)
            canvas = ndimage.grey_dilation(canvas, size=(k, k, 1))
        base = self.img[:, :, ::-1]
        self.composite = base.copy()
        if hit.any():
            self.composite[hit] = (0.15 * base[hit] + 0.85 * canvas[hit]).astype(np.uint8)
        self.im.set_data(self.composite)

        path, n = self.write_colored()
        deg, mm = STEPS[self.step_i]
        d = self.d
        self.hud.set_text(
            f"frame {self.i+1}/{len(self.dirs)}  {os.path.basename(self.dirs[self.i])}"
            f"    step[{self.step_i+1}] = {deg}deg / {mm*1000:.0f}mm    dot = {self.size}\n"
            f"  yaw={d['yaw']:+.3f} deg     pitch={d['pitch']:+.3f} deg     roll={d['roll']:+.3f} deg\n"
            f"  dx={d['dx']*1000:+.1f} mm    dy={d['dy']*1000:+.1f} mm    dz={d['dz']*1000:+.1f} mm\n"
            f"  t = {np.round(self.t0 + np.array([d['dx'], d['dy'], d['dz']]), 5).tolist()}\n"
            f"  colored.ply -> {path}  ({n} points colored)\n"
            f"  w/s dy   a/d dx   q/e dz  |  i/k pitch  j/l yaw  u/o roll  |  1/2/3 step  [ ] dot  r reset  p save")
        self.fig.canvas.draw_idle()

    def on_key(self, ev):
        k = ev.key
        deg, mm = STEPS[self.step_i]
        moved = True
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
        elif k == "]": self.size = min(6, self.size + 1)
        elif k == "[": self.size = max(0, self.size - 1)
        elif k == "r":
            for kk in self.d: self.d[kk] = 0.0
        elif k == "p": self.save()
        elif k == "right": self.i = (self.i + 1) % len(self.dirs); self._load_frame()
        elif k == "left": self.i = (self.i - 1) % len(self.dirs); self._load_frame()
        elif k == "escape": plt.close(self.fig); return
        else: moved = False
        if moved:
            d = self.d
            print(f"[{os.path.basename(self.dirs[self.i])}] yaw={d['yaw']:+.3f} pitch={d['pitch']:+.3f} "
                  f"roll={d['roll']:+.3f} | dx={d['dx']*1000:+.0f} dy={d['dy']*1000:+.0f} "
                  f"dz={d['dz']*1000:+.0f} mm", flush=True)
        self.update()

    def save(self):
        R = self.R0 @ delta_matrix(self.d)
        t = self.t0 + np.array([self.d["dx"], self.d["dy"], self.d["dz"]])
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        yaml.safe_dump({"lidar_to_camera": {"rotation_matrix": R.flatten().tolist(),
                                            "translation": t.tolist()},
                        "note": "tuned by tune_extrinsics_gui.py"},
                       open(self.out_path, "w"), sort_keys=False)
        print("\n=== 微调结果（可直接报给助手）===")
        print(f"  旋转: yaw={self.d['yaw']:+.3f}°  pitch={self.d['pitch']:+.3f}°  roll={self.d['roll']:+.3f}°")
        print(f"  平移: dx={self.d['dx']*1000:+.1f}mm  dy={self.d['dy']*1000:+.1f}mm  dz={self.d['dz']*1000:+.1f}mm")
        print(f"  t = {np.round(t, 5).tolist()}")
        print(f"  已写: {self.out_path}")
        col_path, col_n = self.write_colored(full=True)
        print(f"  彩色云(全量): {col_path}  上色 {col_n} 点")
        col_path = os.path.join(os.path.dirname(self.out_path) or ".", "colored_tuned.ply")
        print(f"  彩色云: {col_path}")
        if self.apply:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy("config/camera_extrinsics.yaml", f"config/extrinsics_history/{stamp}_before_tune.yaml")
            shutil.copy(self.out_path, "config/camera_extrinsics.yaml")
            print("  已装入 config/camera_extrinsics.yaml ✓（旧版已归档）")

    def run(self):
        import signal
        stop = {"v": False}

        def handler(signum, _frame):
            stop["v"] = True
            print(f"\n收到信号 {signum} → 正在退出…（如终端异常, 执行 stty sane）", flush=True)
            plt.close(self.fig)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt → 退出（如终端异常, 执行 stty sane）")


def main() -> int:
    ap = argparse.ArgumentParser(description="键盘微调外参（点云叠在照片上）")
    ap.add_argument("pose_dirs", nargs="*", help="不给则自动选最新+点数最多的帧")
    ap.add_argument("--extrinsics", default="config/camera_extrinsics.yaml")
    ap.add_argument("--camera-info", default="config/camera_info.yaml")
    ap.add_argument("--photo-angle", type=float, default=90.0)
    ap.add_argument("--out", default="/tmp/extrinsics_tuned.yaml")
    ap.add_argument("--max-points", type=int, default=250000, help="参与渲染的点数上限")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    dirs = a.pose_dirs if a.pose_dirs else [pick_best_frame()]
    Tuner(dirs, a.extrinsics, a.camera_info, a.photo_angle,
          a.out, a.max_points, a.apply).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
