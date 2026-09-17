# 相机–LiDAR 外参标定方法（以代码实现为准）

本文档描述项目中 3D 点云投影到 2D 图像所依赖的**相机–LiDAR 外参**
`p_camera = R · p_lidar + t` 是如何确定的。内容全部以仓库代码实现为准。

涉及文件：

| 文件 | 角色 |
|---|---|
| `calibrate_camera_lidar.py` | **自动标定**（外参求解器） |
| `align_camera_lidar.py` | **手动标定**（交互式人工对齐） |
| `colorize_pointcloud.py` | 外参的**消费方**（3D→2D 投影上色） |
| `config/camera_info.yaml` | 相机内参 K + 畸变系数（`save_camera_info.py` 保存，`/camera/color/camera_info`） |
| `config/camera_extrinsics.yaml` | 外参输出文件 |

---

## 1. 自动标定：`calibrate_camera_lidar.py`

### 1.1 总体思路：平面约束对应（plane-correspondence calibration）

不直接匹配 3D–2D 特征点，而是在**多个标定板姿态**下分别测出"同一块棋盘格平面"在两个坐标系中的表达，再用法向量对应求旋转 R、用平面位置对应求平移 t：

- 每个 pose 目录需含 `color.png`（拍照角度下的图像）和 `merged.ply`（点云）。
- **注意参考系**：`merged.ply` 以转台 0° 为基准，而照片在 ready 角度（默认 90°）拍摄。代码把 LiDAR 平面先绕 Z 旋转 `--photo-angle`（默认 90°）转到拍照参考系，再与相机平面对应（`process_pose()`）。

### 1.2 相机侧：棋盘格 PnP 求相机系平面

`detect_camera_plane()`：

1. **角点检测**：`cv2.findChessboardCorners`，图案 4×6 内角点（`BOARD_COLS=4, BOARD_ROWS=6`），标志 `CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE`；失败则对自适应阈值图再用 `findChessboardCornersSB` 兜底。
2. **亚像素精化**：`cv2.cornerSubPix`（5×5 窗口，30 次迭代 / 0.001 eps）。
3. **位姿解算**：`cv2.solvePnP(object_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)`。物体点为 4×6 网格、间距 `SQUARE_SIZE_M = 0.12 m`（板面 600×840 mm）。
4. **平面提取**：棋盘格位于物体系 z=0 平面，所以相机系平面为
   - 法向量 `n_c = R[:, 2]`（PnP 旋转矩阵第三列）
   - 偏置 `d_c = n_c · t`（点到原点有符号距离）
   - 若 `d_c < 0` 则整体取反，保证法向量朝相机（与 LiDAR 侧符号约定一致）。
5. 同时输出板中心 `center_c = R · center_obj + t`（后续求平移用）和板四角投影多边形 `board_polygon`（后续图像筛选用）。

### 1.3 LiDAR 侧：RANSAC 拟合标定板平面

`extract_lidar_board()`（棋盘格是**独立竖立**的，直接取最大平面会找到墙，所以加了裁剪链）：

1. **ROI 裁剪**：`r = √(x²+y²+z²) < 3.0 m` 且 `|atan2(y, −x)| < 20°`（`crop_roi()`）。
2. **地面估计**：对 ROI 点的 z 值做 200 bin 直方图，取峰值作为地面 z（`detect_ground_z()`）。
3. **高度过滤**：保留离地高度在 **(0.4 m, 2.6 m)** 的点（floor = z 峰值），剔除地面/天花板/杂物。
4. **RANSAC 平面**：`ransac_plane()` —— 每轮随机采 3 点求法向量，内点阈值 **0.02 m**，迭代 **3000 次**，取内点最多者。
5. **SVD 精拟合**：对内点用 `fit_plane_svd()` 重新拟合——去质心后 SVD，**最小奇异值对应的右奇异向量**即平面法向量（最小二乘意义下的最优平面）。

得到 LiDAR 系平面 `(n_l, d_l)`，以及内点点集（其质心 `center_l` 后续求平移用）。

### 1.4 全局求解：`solve_all()`

#### 旋转 R —— 法向量正交 Procrustes（Kabsch/SVD）

`solve_rotation()`：

```
H = Σᵢ n_lᵢ · n_cᵢᵀ          （3×3 协方差矩阵）
U, S, Vᵀ = svd(H)
d = sign(det(VᵀUᵀ))
R = V · diag(1, 1, d) · Uᵀ
```

即在两组单位法向量间求**最优正交旋转**，`diag(1,1,d)` 保证 det(R)=+1（纯旋转而非含反射）。

**可观测性**：单个平面法向量只约束 2 个旋转自由度（绕法向的旋转不可观），所以 **R 至少需要 2 个不平行板位姿**（代码未显式检查，输入不足会得到错误解）。

#### 平移 t —— 板质心差平均（注意：与 AGENTS.md 描述不同）

```python
center_l = LiDAR 平面内点质心
center_c = PnP 板中心（相机系）
t = mean_i( center_cᵢ − R · center_lᵢ )      # 对多个 pose 取平均
```

> ⚠️ **与 AGENTS.md 的差异**：AGENTS.md 写的是 `d_c − d_l = n_c · t` 最小二乘，
> 但当前代码实际用的是**板中心（质心）差平均**。质心法把整块板的 LiDAR 内点质心
> 当作对应点，因此对"板上混有共面杂点（墙点超出板范围会拉偏质心）"更敏感——
> 这正是下一步图像引导精修存在的原因。以代码为准。

### 1.5 图像引导精修：`refine_pose_plane()`

初始 (R, t) 解出后，做一次精修抑制"墙面共面点污染"：

1. 用当前 (R, t) 把 LiDAR 内点投影到图像（`cv2.projectPoints`）。
2. 用 `cv2.pointPolygonTest` 只保留**落在棋盘格四角投影多边形内（外扩 12 px）**的点（`dbg_4_board_selected.ply`）。
3. 对筛选后的点重新 `fit_plane_svd()`，更新该 pose 的 LiDAR 平面与质心。
4. 用精修后的所有 pose 重新 `solve_all()`。

若筛选后点数 < 50，则放弃精修保留原平面（防止误删）。

### 1.6 验证与输出

- **法向量角误差**：`arccos(n_c · (R·n_l))`（度），输出 raw / refined 两组及 RMS。
- **板中心残差**：`‖(c_c − R·c_l) − mean‖`，反映平移一致性。
- **诊断**：`R · [0,0,1]` 应 ≈ 相机图像"上"方向 `[0,−1,0]`。
- **可视化**：每 pose 输出 `validation_overlay.png`（红点 = 投影的 LiDAR 内点，绿圈 = 图像角点），直接肉眼看对齐质量。
- **输出**：`config/camera_extrinsics.yaml`（`--output` 可改）：

```yaml
lidar_to_camera:
  rotation_matrix: [...]   # 3×3 按行展平，共 9 个数
  translation: [...]       # 3 个数
note: "p_camera = R @ p_lidar + t (merged-frame LiDAR; see AGENTS.md)"
```

使用方式：

```bash
python3 calibrate_camera_lidar.py POSE_DIR [POSE_DIR ...] \
    --camera-info config/camera_info.yaml --photo-angle 90
```

（`--seed 0` 固定 RANSAC 随机种子，结果可复现；`--photo-angle` 默认 90。）

---

## 2. 手动标定：`align_camera_lidar.py`

纯人工方法，无优化算法：实时把点云投影叠加到图像上，人用键盘微调 6 自由度直到对齐。

- **投影模型**（`project_points()`）：`p_cam = R·p + t`，`uv = K·p_cam / z`，仅显示 `z > 0.1 m` 的点；点按深度做 JET 伪彩。
- **初始值**：`t = [0, 0, 1]`，`R = I`；点云 > 10 万时随机降采样到 10 万。
- **键盘调节**（步长 `+`/`-` 倍增/倍减，默认 0.01 m / 0.01°）：
  `q/a` tx、`w/s` ty、`e/d` tz、`r/f` rx、`t/g` ry、`y/h` rz（欧拉角增量左乘：`R = ΔR · R`）。
- **保存**：按 `S` 存 `<data_dir>/config/camera_extrinsics.yaml`。

> ⚠️ **格式差异**：手动工具保存的是**顶层** `translation / rotation_matrix / euler_deg`，
> 且路径是 `<data_dir>/config/`；而 `colorize_pointcloud.load_extrinsics()` 期望的是
> **`lidar_to_camera` 包裹的格式**（即自动标定的输出）。手动保存的文件不能直接被
> 上色脚本读取，需要手动调整 YAML 结构（或改用自动标定的输出文件）。

---

## 3. 外参的消费：3D→2D 投影 `colorize_pointcloud.py`

外参确定后的实际用途——把 `merged.ply` 投影到 `color.png` 取颜色：

1. 读外参 (R, t) 与内参 (K, dist)。
2. **参考系修正**：合并点云以转台 0° 为基准，外参是拍照系下的，所以先左乘
   `R_full = R @ Rz(photo_angle)`（默认 90°，与标定侧的 `--photo-angle` 对应）。
3. `cv2.projectPoints(points, Rodrigues(R_full), t, K, dist)` 投影（含畸变）。
4. **可见性判断**：投影值有限、`|uv| < 1e6`、`cam_z > 0`（相机前方）、像素在图像范围内。
5. **最近邻取色**：`colors = image[v, u]`（BGR→RGB）；不可见点置灰 (30, 30, 30)。
6. 输出 RGB `colored.ply`。

`turntable_gui.py` 在每次扫描 merge 后自动调用它（`_colorize_merged()`），
前提是仓库根目录存在 `config/camera_extrinsics.yaml`。

---

## 4. 方法对比与适用场景

| | 自动标定 (`calibrate_camera_lidar.py`) | 手动对齐 (`align_camera_lidar.py`) |
|---|---|---|
| 原理 | 多姿态棋盘格平面约束：PnP + RANSAC + Kabsch + 质心平均 + 图像精修 | 人工目视对齐 |
| 输入 | ≥2 个 pose 目录（`color.png` + `merged.ply`） | 1 个数据目录 |
| 精度 | 取决于角点/平面拟合质量，有定量残差报告 | 依赖人眼，无量化指标 |
| 输出格式 | `lidar_to_camera` 包裹（可直接被上色脚本用） | 顶层字段（需手动适配） |
| 适用 | 正式标定、可复现 | 快速验证、无棋盘格时兜底 |

## 5. 已知注意事项（以代码为准）

1. **平移是质心差平均**（见 1.4），不是平面偏置最小二乘；共面墙点会拉偏质心，靠 `refine_pose_plane()` 的图像多边形筛选缓解。
2. **旋转至少需要 2 个不平行板位姿**；实践中建议 ≥3 个、姿态分布尽量展开。
3. **法向量符号**两侧都被规范化到 `d ≥ 0`（朝相机），两侧约定一致才能正确对应。
4. `--photo-angle` 在标定（转 LiDAR 平面）和上色（转点云）两侧**必须一致**，否则投影系统性偏移。
5. RANSAC 用固定种子（`--seed 0`），结果确定；换种子可作为稳定性检查。
6. `solvePnP` 使用 `SOLVEPNP_IPPE`（平面目标专用，返回两个解中的第一个，代码取 `ok` 即接受的解）。
