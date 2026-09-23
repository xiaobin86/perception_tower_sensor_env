# AGENTS.md — Perception Tower

Project-specific conventions and memory for coding agents. Keep this file updated.

## Point cloud / coordinate conventions (IMPORTANT)

For merged point clouds produced by the turntable pipeline (`merged.ply`):

- **Z is the vertical axis** (the turntable rotation axis is vertical).
- **XY is the horizontal plane** (the turntable rotation plane).
- The static scene / calibration board sits in the **−X** direction (~2 m from the LiDAR origin).
- Horizontal angle around the LiDAR origin: `theta = atan2(y, -x)` (0 = −X direction, positive toward +Y).
- LiDAR origin is `(0, 0, 0)` in the merged frame; the floor is at `z ≈ −1.65 m` in the current setup.

### Standard crop (region of interest)

To isolate the working region (used for calibration / board search):

```
r = sqrt(x^2 + y^2 + z^2) < 3.0 m
|atan2(y, -x)| < 20 deg
```

This keeps ~36% of a full scan (e.g. 108,114 / 298,519 in the `20260910_055020` capture).
Reusable tool: `crop_pointcloud.py`.

## Calibration target

- Checkerboard (2026-09 换新板): **9×7 squares = 8×6 inner corners**, rectangular squares
  **88.75 mm (X) × 85.5 mm (Y)** (board ≈ 799×599 mm). 旧板（5×7 格 / 4×6 角点 / 标称 120mm
  实测 116mm）因格距不标准已退役 —— 新板首次标定前**务必用尺子复核格距**（血泪见
  标定经验教训）。
- Camera: Orbbec Gemini 336L, color 1280×720, intrinsics in `config/camera_info.yaml` (plumb_bob distortion).
- Goal: LiDAR→color-camera extrinsic via plane correspondences from **≥3 non-parallel board poses**
  (`n_c = R·n_l`, `d_c − d_l = n_c·t`).

## Gotchas

- **open3d 已固化进镜像**：`segment_board.py` / `visualize_ply.py` / `lakibeam_viewer.py`
  依赖 open3d，Dockerfile 通过清华镜像源安装（层缓存命中后重建秒过）。若镜像被清需
  手动补：`pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple`。

- `merged.ply` is merged with the turntable reference at angle **0**, but the camera photo is taken at the
  ready angle (**90°**). For camera–LiDAR calibration the merge must be re-referenced to the photo angle:
  transform each frame by `(angle − photo_angle)` instead of `angle`, so the merged cloud is in the LiDAR
  frame at the photo pose and is directly comparable to the image.
- Camera and LiDAR are rigidly coupled; rotating the turntable changes the common viewpoint only, so the
  extrinsic is constant.
- **Never `kill -9` the camera nodes.** `kill -9` leaves the Orbbec USB device stuck
  (`usbfs: did not claim interface 0 before use` → `Device response size(0)`), and recreating the container
  does NOT reset it. Use graceful SIGINT/SIGTERM. Recovery without rebuild: reset the `2bc5` USB device with
  `USBDEVFS_RESET` (ioctl `0x5514`) on `/dev/bus/usb/<bus>/<dev>`.
- The container maps `/dev` as a private tmpfs; a full USB re-enumeration (new devnum) will not appear
  automatically — prefer `USBDEVFS_RESET` (keeps the address) over unbind/rebind.

## Free-standing board segmentation (calibration)

The board is **free-standing** (not flush on the wall), so plain "largest plane" finds the wall, not the board.
Working recipe:

1. Crop as above (`r < 3 m`, `|atan2(y, −x)| < 20°`).
2. Keep only points whose **height above the floor** is in `(0.4 m, 2.6 m)`. Floor is at `z ≈ −1.648 m`,
   so that is `−1.248 < z < 0.952`. This drops floor/ceiling and low/high clutter.
   (`0.4 m` is the tuned lower bound; `0.6 m` clips the bottom of the board.)
3. RANSAC the largest plane → it is now the board (vertical, normal ≈ ±X, extent ≈ board size).

Verified on `20260910_055020`: board plane `n = [−0.969, 0.225, −0.101]`, `d = −1.826` (plane `n·p + d = 0`),
centroid `[−1.83, −0.07, −0.72]`, ~24.7k inliers; the dense board patch is ~0.6 m × 0.84 m (a few coplanar
wall points just beyond the board also fall on the plane).
Outputs: `cropped_3m_20deg_board.ply`, `view_board.png`.

## Camera-LiDAR calibration solver

`calibrate_camera_lidar.py POSE_DIR [POSE_DIR ...]` — each `POSE_DIR` holds `color.png` + `merged.ply`
(the merged cloud must be referenced to the photo angle, see gotchas). Per pose it does:

1. **camera side** → multi-strategy checkerboard detection + `cornerSubPix` →
   `solvePnP(SOLVEPNP_IPPE)` + `solvePnPRefineLM` → camera plane `(n_c, d_c)` with **robust corner
   rejection** (`solve_pnp_robust`): corners with reprojection error > 2.5 px are dropped and PnP re-solved
   (one bad corner can bias the whole pose by ~0.5°/9 mm — measured); a frame with > 2 bad corners is
   rejected entirely;
2. **lidar side** → `segment_board.py extract_rect_plane`: remove ground (RANSAC horizontal) →
   deterministic multi-seed RANSAC candidates (±2 cm, `DIST_THR`) → per-candidate SVD refine →
   2D sliding of the known 0.84×0.60 m window on the plane (score = in-window − 3×ring, ring = 3 cm
   band outside the window) → largest connected component of the plane band (the ±2 cm slab is
   infinitely extended and would otherwise swallow floor/ceiling/scatter) → final mask on the original
   cloud → quality gate (frame skipped if: fitted size < 0.78/0.54 m, size error > 16 %, or
   selected/band ratio < 0.75). All thresholds are named constants at the top of `segment_board.py`;
3. over all poses solve `n_c = R·n_l` (Kabsch/SVD) and `t` from board-center differences, then
   image-space translation refinement (R fixed; t.x/t.y only).

Constraints: **R needs ≥2 non-parallel board planes, t needs ≥3** (one plane fixes 3 of 6 DOF).
Intrinsics from `config/camera_info.yaml` (distortion zeroed — the color image is treated as already
rectified; pre-zeroing coefficients are in `config/camera_info.yaml.bak_withdist` and the pipeline
passes `d` through `cv2.projectPoints`, so restoring them needs no code change); board is 8×6 inner
corners, rectangular 88.75×85.5 mm squares (SQUARE_SIZE_X/Y in `calibrate_camera_lidar.py`).

**Official extrinsics (current — 2026-09-24, MATLAB 标定版, 用户指定)** =
MATLAB `lidarCameraTform`（用户 2026-09-24 解算，**4 位有效数字**）换算到照片位姿约定：
`R = R_matlab·Rz(−90°)`（角度 0 参考系 → 照片位姿参考系），`t = [0.0263, 0.0590, −0.0198]`。
原始 MATLAB 矩阵与换算说明存于 `config/camera_extrinsics.yaml` 的 `matlab_source` 字段。
注意：(1) 源内参为 MATLAB 拟合值（fx 610.75 / fy 625.63 / cx 652.81 / cy 373.98），渲染仍用
`config/camera_info.yaml`（fy 611.375），两者非同一内参组；(2) 4 位有效数字引入的量化
约 ±0.05 mm/±0.05° 量级误差。所有 `20260923_*` 的 colored.ply 与 dense_colored.ply 已用此版刷新；
`colored_mlab.ply` 为同参数的旁路对照（photo=0 直渲，内容应与 colored.ply 等价）。

**Official extrinsics (2026-09-23 Python 重解版, SUPERSEDED, archived)** = raw solve over the
11 good `20260923_*` poses (19 captured; 070238 no board plane, 070700/070957/
072937 no checkerboard, 070807/071405/071705 sparse board, 073136 >2 bad corners), new board
(88.75×85.5 mm), fy = 611.375 factory, distortion zeroed: `t = [0.0264, 0.0734, 0.0317]`,
R = solve, **no tweaks**. Diagnostics: up = [0.01, −0.818, −0.576], |Δ| = 14.0 px,
normal-angle err rms 0.68°, 落格率 97–100%. Key fact: the user confirmed the camera pitch was
physically adjusted between the 0918 era and 0923 — this explains the ~12° pitch difference vs
the 0918-era extrinsics. Archived:
`config/extrinsics_history/20260924_021045_0923solve_before_matlab.yaml`.
深度云审计 (audit_tz_extrinsics.py, 对上一版 MATLAB t.z=−0.078): 此版胜
(加权 RMS 27 vs 40 mm, 去偏 std 18 vs 38 mm, 22/24 帧) — 注意被审计的 MATLAB 版
已非当前 MATLAB 版 (t.z 已改 −0.0198)。

**0918-era extrinsics (SUPERSEDED, archived)** = solve over the 16 good `20260918_*` poses
(073749 wide-scan excluded, 062814 auto-skipped: no checkerboard),
SQUARE_SIZE_M=0.116 实测格距, plus the keyboard eye-validated roll tweak (R = solve·Ry(roll),
tune_rotation_live.py): `t = [0.0293, 0.0556, −0.0140]`, R = solve + **roll −0.9°**. Identical copy:
`config/extrinsics_history/20260921_070000_before_recalib.yaml`. Only valid for data captured
before the camera-pitch adjustment.

**2026-09-21 re-solve (REJECTED, archived)** = raw solve over 17 good `20260921_*` poses
(different-distance set; 042647/042731 no checkerboard, 060302/060448 board size gate), new standard
board (88.75×85.5 mm), original fy, distortion zeroed: `t = [0.0235, 0.0849, 0.0187]`,
R = solve, no tweaks. Re-solved the same evening: reproduced the morning result **bit-exactly**
(deterministic pipeline, healthy diagnostics: up ≈ [0, −0.91, −0.41], |Δ| ≈ 12 px).
Rejection judge: multi-frame full-scene eye matrix on 085156 — 2×2 {this solve, 116} ×
{fy 590.59, fy 611.375}; **116+fy611 won**. Why the quantitative judges could not arbitrate:
board-overlap is structurally blind to fy (camera-side PnP quad reuses the same K, errors cancel)
and insensitive to 0.74°; depth-cloud Δt.y neutral for both (±1.5 mm); depth-cloud pitch residual
(+1.7~3.4° for this solve, ≈0 for 116) **fails the falsification test** (RMS 23→42 mm, residual
reappears +3.8° — D2C parallax systematic, same signature as the historic +3.86°). Open lead:
user's ruler says new-board X pitch is 88.5 mm (code/this file: 88.75, +0.28%) — worth ~6 mm of
Δt.x, not the main term of the 44 mm. Archived:
`config/extrinsics_history/20260921_220535_20260921_solve_rejected.yaml`.
Diff vs current: ΔR = 0.74° (mainly pitch +0.65°), Δt = [−5.8, +29.3, +32.7] mm, concentrated in
the board-weak t.y/pitch subspace. NOTE (2026-09-23 update): the "install unchanged since 0918 —
the rig did not move" claim in the original text is **superseded** — the user confirmed the camera
pitch was physically adjusted, so this solve's deviation is no longer evidence against it; the 21号
eye-check rejection stands only for the pre-adjustment era.

### Version-selection logic (why this exact extrinsic is official)

The selection is evidence-chained; each step's judge is listed:

1. **Input hygiene first**: camera-side bad-corner rejection (070027 had a single 44.5 px corner that
   dragged its whole pose; 061943/073031 similar) + lidar-side cleaned band + quality gate. A joint
   solve is only as good as its per-pose observations — equal-weight outliers were the historical
   root cause of diverged solves (including a 180° mirror flip traced to random RANSAC normal signs;
   the convention is now `d > 0`, normal pointing away from the sensor, matching the camera side).
2. **Solve health**: accept only if `diagnostic: R@merged_up(Z) ≈ [0, −0.92, −0.39]` (camera up in the
   merged frame) and image residual `|Δ| ≈ 12 px`.
3. **t.y (vertical) — weakly observable from boards** (plane normals stay nearly coplanar for upright
   boards). Judge: the camera **depth cloud** (independent sensor), not the LiDAR board fit. The
   depth-based solve (`solve_ty_residual.py`) returned Δt.y = −0.3 mm at dy −30 mm (noise level). Offsets dy −30/−20 mm and raw
   were all eye-reviewed; the board-only vertical stays weakly observable, so the final
   vertical/depth values come from keyboard eye-tuning on the last frame (ty +30 mm, tz −50 mm
   on top of the solve); vertical is closed.
4. **Rotation tweaks must survive a falsification test**: the depth cloud persistently suggested
   +3.86° pitch about world X, but applying it *worsened* the depth RMS (40.3 → 42.5 mm) and the
   residual re-appeared (+2.85° more) — the signature of a **parallax/D2C systematic, not a rotation
   error**. Eye check agreed (3.8° looked too big). Rejected; no rotation fudge factors in the
   official version. (`solve_pitch_residual.py` documents the experiment.)
5. **Cross-validation of the final pick**: the cleaned-pipeline re-solve (v2) differs from the
   previous official by only ΔR = 0.35°, Δt ≈ [3, −3, 4] mm — two independent pipelines converging
   is the strongest available confirmation short of a ground-truth rig.

Pose diversity beats pose count: boards should be **tilted** (pitch ±20…45°) and spread in azimuth.
All-upright boards leave the camera pitch/roll weakly observable and their incidence-angle bias does
not self-correct. Judge candidate extrinsics against the depth cloud — board-plane RMS alone is
misleading (and even the depth cloud can be fooled by parallax systematics, see step 4).

## 标定经验教训（20260918 数据集排障 3 天，代价：tz 偏差 50mm）

1. **LiDAR/相机等设备的刚性安装位姿必须准确**：安装角、偏心、to_world 映射的任何偏差
   都会以系统误差形式进入外参；安装配置（`install_config.py`）改动后必须重标。
2. **标定板必须标准**：标称 120mm 的格子实测 116mm（−3.3%）→ PnP 测距同比例偏大 →
   被外参 t 吸收 → 深度方向差 5cm。**尺度吸收使一切角点/重投影类判据失效**（PnP 把
   尺度误差全吸收进距离，重投影 RMS 依然 <1px）——这次所有自动判据都没报警，最后是
   深度云对比 + 人眼抓出来的。**教训：标定前先拿尺子量格子**。
3. **诊断方法论**：双传感器"同原点、同物点"测距对比（`solve_range_bias.py`：
   PnP 板深度 vs Orbbec 深度云）能暴露一切被 t 吸收的系统偏差；它不经外参 t，
   是标定后的第一道审计。预测-验证闭环：理论预测 tz 吸收 +54mm ≈ 人眼微调 −50mm；
   改 `SQUARE_SIZE_M=0.116` 后裸解与全套人工微调版差 <4mm —— 根因确认。
4. 完整推导与配图：`docs/tz_bias_analysis.md` + `docs/fig1~4_*.png`（`make_tz_figs.py` 生成）。
5. **被目测评判否掉的修正严禁回写配置**：2026-09-21 上午四组合目测已否掉 fy×0.966，当日
   14:48 它仍被写进 `config/camera_info.yaml`（fy 590.59），而外参是 07:00 在 fy=611.375 下
   解算的 —— 14:48 之后所有"当前外参"渲染都与求解内参不自洽（垂直方向 ~3.4% 投影畸变，
   边缘约 20px），当天大批对比结论被污染。规则：判定/渲染用内参与求解用内参必须一致；
   改内参后必须重解外参再对比。
6. **单帧目测选版不可靠**：9-21 上午仅凭 064431 单帧四选一就采纳了 21 号裸解；当晚用
   085156 实际场景做 2×2 多帧对比即被推翻（116+fy611 胜）。外参选版判决必须用多帧、
   实际采集场景（最好含不同距离/朝向），单帧标定位姿上的差异常常展不开。

## Existing helper scripts

- `save_camera_info.py` — dump `/camera/color/camera_info` to `config/camera_info.yaml`.
- `undistort_images.py` — undistort color images.
- `visualize_ply.py` — point cloud viewer.
- `crop_pointcloud.py` — crop by distance + horizontal angular wedge (see conventions above).
- `calibrate_camera_lidar.py` — automatic checkerboard camera-LiDAR calibration (see above).
- `segment_board.py` — LiDAR board segmentation (config constants at top; `--save` writes
  `board_rect.ply` + `board_rect_overlay.ply`: red = selected, blue = in-band not selected).
  `turntable_gui.py` runs it automatically after every scan merge (`_detect_board`).
- `annotate_camera_side.py` — per-frame camera-side annotation (corners / reprojection / board quad /
  center) + bad-corner statistics over any frame list; uses the same rejection policy as calibration.
- `solve_ty_residual.py` / `solve_pitch_residual.py` — depth-cloud residual solvers for Δt.y and
  pitch (use for measurement; verify before applying, see version-selection logic step 4).
- `audit_tz_extrinsics.py` — depth-cloud head-to-head arbitration between two candidate
  extrinsics (official 0923 vs MATLAB lidarCameraTform, photo=0 frame): per-frame robust RMS of
  z_lidar − D_depth; 2026-09-23 result: official wins (weighted RMS 27 vs 40 mm, bias-free std
  18 vs 38 mm, 22/24 frames).
- `verify_board.py` — project the segmented board into the photo and score quad overlap with the
  detected checkerboard (camera-as-reference frame audit).
- `run_calibration.sh` — recompute the official extrinsics from every `turntable_output/calib_*` into
  `config/camera_extrinsics.yaml`.
- `colorize_pointcloud.py` — project `merged.ply` onto `color.png` and write an RGB `colored.ply`
  (nearest-neighbour pixel sampling; points outside the image stay gray). `turntable_gui.py` calls it
  automatically after each scan merge when `config/camera_extrinsics.yaml` exists (both the LakiBeam and
  the Fairy path; photo angle = `(scan_start + scan_end) / 2`).

## Scan range filter (turntable_gui)

`Range (min, max)` in the GUI (or `--min-dist` / `--max-dist`, default `0.2, 3.0` m) drops merged points
outside `min < r ≤ max`; it applies to the LakiBeam merge, the Fairy merge and the depth-cloud crop.
Separator is the half-width comma only. `min = 0.2 m` removes the LiDAR's near-field self-returns
(mount/bracket at ~5 cm) that otherwise dominate the merged cloud.
