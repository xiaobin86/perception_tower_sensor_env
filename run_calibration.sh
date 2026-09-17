#!/usr/bin/env bash
# 用 turntable_output/calib_* 的标定数据集重算正式外参 -> config/camera_extrinsics.yaml
# 用法: ./run_calibration.sh
set -euo pipefail
cd "$(dirname "$0")"

shopt -s nullglob
dirs=(turntable_output/calib_*)
if [ ${#dirs[@]} -eq 0 ]; then
    echo "未找到标定数据集 turntable_output/calib_*" >&2
    exit 1
fi

echo "标定数据集: ${#dirs[@]} 个"
python3 calibrate_camera_lidar.py "${dirs[@]}" --output config/camera_extrinsics.yaml
