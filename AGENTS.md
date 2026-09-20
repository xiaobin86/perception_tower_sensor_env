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

- Checkerboard: **5×7 squares = 4×6 inner corners**, square size **120 mm** (board 600×840 mm).
- Camera: Orbbec Gemini 336L, color 1280×720, intrinsics in `config/camera_info.yaml` (plumb_bob distortion).
- Goal: LiDAR→color-camera extrinsic via plane correspondences from **≥3 non-parallel board poses**
  (`n_c = R·n_l`, `d_c − d_l = n_c·t`).

## Gotchas

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
passes `d` through `cv2.projectPoints`, so restoring them needs no code change); board is 4×6 inner
corners, 120 mm squares.

**Official extrinsics (2026-09-20)** = solve over the 16 good `20260918_*` poses (073749 wide-scan
excluded, 062814 auto-skipped: no checkerboard), plus the eye/deepth-validated `t.y` offset (see
version-selection logic below): `t = [0.02917, 0.030, 0.03994]`,
   R = solve + **roll −0.9°** (raw solve plus eye-validated dy −30 mm and roll; applied via
   `tune_rotation_live.py` keyboard tuning). Earlier backups were removed on
purpose; `run_calibration.sh` still rebuilds from `turntable_output/calib_*` when those exist.

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
   were all eye-reviewed; **final pick: dy −30 mm** (the depth cloud also peaks there);
   vertical is closed.
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
