# Perception Tower

基于 ROS2 Humble 的感知塔项目，集成 Orbbec Gemini 336L 相机和 RoboSense Fairy LiDAR。

## 快速开始

### 1. 启动 Docker 容器

```bash
cd /mnt/d/work/perception-tower
docker compose -f docker/docker-compose.yml up -d
docker exec -it perception_tower bash
```

或在 VS Code 中：
- `Ctrl+Shift+P` → `Dev Containers: Rebuild Container`

### 2. 启动传感器驱动

**终端 1 - RoboSense Fairy LiDAR：**
```bash
ros2 launch rslidar_sdk start.py
```

**终端 2 - Orbbec Gemini 336L 相机：**
```bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

### 3. 验证话题发布

```bash
ros2 topic list
ros2 topic hz /fairy/points
ros2 topic hz /camera/color/image_raw
```

### 4. 一键启动所有传感器（可选）

如果需要同时启动 LiDAR、相机和转台节点，可以使用一键启动命令：

```bash
ros2 launch perception_tower_sensor sensor_env.launch.py
```

指定转台串口：

```bash
ros2 launch perception_tower_sensor sensor_env.launch.py turntable_port:=/dev/ttyUSB1
```

这个命令会自动启动：
- RoboSense Fairy LiDAR
- Orbbec Gemini 336L 相机
- 转台控制节点

## 环境说明

容器内已预装：
- ROS2 Humble
- Orbbec SDK ROS2 (v2-main)
- RoboSense rslidar SDK (v1.5.19)

环境变量已自动配置（写入 `~/.bashrc`）：
- `ROS_DOMAIN_ID=0`
- `ROS_LOCALHOST_ONLY=0`
