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

1. image → multi-strategy checkerboard detection (adaptive-threshold window variants / Otsu / B-G
   channel, both 4×6 and 6×4) + `cornerSubPix` → `solvePnP(SOLVEPNP_IPPE)` → camera plane `(n_c, d_c)`;
2. cloud → remove the floor plane (RANSAC largest near-horizontal plane) + drop self-returns (<0.25 m)
   → RANSAC search for the board plane (iterating large non-board planes away) → keep the largest
   connected component of the plane inliers → trim to the known 0.60×0.84 m board window → `(n_l, d_l)`;
3. over all poses solve `n_c = R·n_l` (Kabsch/SVD) and `t` from board-center differences; then one
   image-guided refinement pass (project the LiDAR inliers into the photo, keep those inside the board quad).

Constraints: **R needs ≥2 non-parallel board planes, t needs ≥3** (one plane fixes 3 of 6 DOF).
Intrinsics from `config/camera_info.yaml`; board is 4×6 inner corners, 120 mm squares.

Calibration datasets live in `turntable_output/calib_*` (the `calib_` prefix marks them; other timestamp
directories are ad-hoc scans). Official extrinsics:

    ./run_calibration.sh    # 用所有 turntable_output/calib_* 重算 -> config/camera_extrinsics.yaml

Pose diversity beats pose count: boards should be **tilted** (pitch ±20…45°) and spread in azimuth.
All-upright boards leave the camera pitch/roll weakly observable (measured plane normals stay nearly
coplanar) and their incidence-angle bias does not self-correct, so an upright-only fit can look
self-consistent (~1°) yet be several degrees off; mixing many upright poses into a joint fit degrades it.
Judge candidate extrinsics against the depth cloud (independent sensor) — board-plane RMS alone is
misleading.

## Existing helper scripts

- `save_camera_info.py` — dump `/camera/color/camera_info` to `config/camera_info.yaml`.
- `undistort_images.py` — undistort color images.
- `align_camera_lidar.py` — manual extrinsics alignment (keyboard) + projection overlay.
- `visualize_ply.py` — point cloud viewer.
- `crop_pointcloud.py` — crop by distance + horizontal angular wedge (see conventions above).
- `calibrate_camera_lidar.py` — automatic checkerboard camera-LiDAR calibration (see above).
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
