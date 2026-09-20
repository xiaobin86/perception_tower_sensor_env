#!/usr/bin/env python3
"""生成 tz 偏差分析文章配图 (docs/fig1~4_*.png)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

D = "docs"

# ---- Fig 1: pinhole ranging principle ----
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([0], [0], "ko", ms=8)
ax.text(0.02, 0.08, "camera center (origin)", fontsize=10)
ax.plot([1.0, 1.0], [-1.6, 1.6], "b-", lw=2)
ax.text(1.02, 1.65, "image plane (px)", color="b", fontsize=10)
ax.plot([4.0, 4.0], [-1.0, 1.0], "g-", lw=4)
ax.text(4.05, 1.1, "board, real width W", color="g", fontsize=10)
for y0, y1 in [(1.0, 1.6), (-1.0, -1.6)]:
    ax.plot([0, 1.0], [0, y1], "k--", lw=0.8)
    ax.plot([0, 4.0], [0, y0], "k-", lw=0.8)
ax.plot([0, 4.35], [0, 0], "k:", lw=0.8)
ax.annotate("", xy=(4.35, 0.02), xytext=(0, 0.02), arrowprops=dict(arrowstyle="<->"))
ax.text(2.1, 0.1, "Z (unknown, to solve)", fontsize=11)
ax.plot([1.0, 1.0], [1.6, 1.75], "b-", lw=2)
ax.annotate("", xy=(1.08, 1.72), xytext=(1.08, -1.72), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(1.12, 0.0, "w (px, measured)", color="b", fontsize=10)
ax.text(2.6, -1.9, "similar triangles:  Z = fy * W / w", fontsize=13,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
ax.set_xlim(-0.4, 5.2); ax.set_ylim(-2.3, 2.2); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(f"{D}/fig1_pinhole_ranging.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: scale absorption (why reprojection cannot see it) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (z, label, col) in zip(axes, [(4.0, "case A: W=0.60 m at Z=4.00 m", "g"),
                                        (4.138, "case B: W=0.62 m at Z=4.138 m", "r")]):
    ax.plot([0], [0], "ko", ms=7)
    ax.plot([1.0, 1.0], [-1.6, 1.6], "b-", lw=2)
    h = 1.0 * (z / 4.0)
    ax.plot([z, z], [-h, h], color=col, lw=4)
    for s in (1, -1):
        ax.plot([0, 1.0], [0, 1.6 * s], "k--", lw=0.8)
        ax.plot([0, z], [0, h * s], "k-", lw=0.8)
    ax.set_title(label, fontsize=10)
    ax.text(z / 2, -2.0, f"same image size w", fontsize=9, ha="center")
    ax.set_xlim(-0.3, 5.0); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Identical pixels  <=>  (W, Z) only determined up to scale:  Z_est = (W_assumed/W_true) * Z_true",
             fontsize=11)
fig.savefig(f"{D}/fig2_scale_absorption.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: measured data ----
names = ["061943", "062116", "062222", "062327", "062453", "062733", "065457",
         "070027", "072600", "072638", "072810", "073115", "073204", "073300", "073336"]
z_pnp = [1.635, 1.627, 1.600, 1.752, 1.740, 1.679, 1.638, 1.623, 1.599, 1.588,
         1.623, 1.602, 1.601, 1.620, 1.616]
z_dep = [1.527, 1.513, 1.483, 1.702, 1.696, 1.620, 1.577, 1.569, 1.573, 1.559,
         1.565, 1.559, 1.553, 1.556, 1.576]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - 0.2, np.array(z_pnp) * 1000, 0.4, label="Z_pnp (PnP, size-based)", color="tomato")
ax.bar(x + 0.2, np.array(z_dep) * 1000, 0.4, label="Z_depth (Orbbec depth, independent)", color="steelblue")
for i in x:
    ax.text(i, max(z_pnp[i], z_dep[i]) * 1000 + 8, f"{(z_pnp[i]-z_dep[i])*1000:+.0f}",
            ha="center", fontsize=8, color="darkred")
ax.axhline(np.median(z_dep) * 1000, color="steelblue", ls="--", lw=1)
ax.text(len(names) - 0.5, np.median(z_dep) * 1000 - 60, f"median $\\Delta$ = +54 mm",
        ha="right", fontsize=11, color="darkred",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, fontsize=8)
ax.set_ylabel("board center Z (mm)")
ax.set_title("Same origin, same physical point, extrinsic-free: PnP depth is ~54 mm farther than depth camera")
ax.legend(); fig.tight_layout()
fig.savefig(f"{D}/fig3_measured_delta.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: error chain ----
fig, ax = plt.subplots(figsize=(11, 3.2))
boxes = [
    ("true board\ngrid = 116 mm", "#d9ead3"),
    ("PnP assumes\ngrid = 120 mm", "#fce5cd"),
    ("Z overestimated\n+3.45% (+54 mm)", "#f4cccc"),
    ("extrinsic t absorbs\nthe +54 mm", "#fce5cd"),
    ("colored render:\ncloud pushed 54 mm\ntoo far", "#f4cccc"),
    ("eye tune\ntz -50 mm\ncompensates", "#d9ead3"),
]
for i, (txt, col) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((i * 1.72, 0.3), 1.45, 1.4, boxstyle="round,pad=0.08",
                                fc=col, ec="k", lw=1))
    ax.text(i * 1.72 + 0.72, 1.0, txt, ha="center", va="center", fontsize=9)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((i * 1.72 + 1.47, 1.0), (i * 1.72 + 1.70, 1.0),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.4))
ax.text(8.6, 0.0, "measured: median(Z_pnp - Z_depth) = +54 mm  ==  -(eye tune -50 mm)", fontsize=10,
        ha="center", color="darkred")
ax.set_xlim(-0.2, 10.6); ax.set_ylim(-0.4, 2.1); ax.axis("off")
fig.savefig(f"{D}/fig4_error_chain.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("4 figures saved to docs/")
