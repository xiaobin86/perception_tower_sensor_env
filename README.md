# Perception Tower

基于 ROS2 Humble 的感知塔项目，集成 Orbbec Gemini 336L 相机、RoboSense Fairy LiDAR 和 STM32 伺服转台。

## 硬件依赖

| 设备 | 型号 | 接口 |
|------|------|------|
| 相机 | Orbbec Gemini 336L | USB 3.0 |
| LiDAR | RoboSense Fairy | 网线（推荐直连宿主机 enp4s0） |
| 转台 | STM32F103C8T6 + YQD-42GS | USB-TTL（/dev/ttyUSB*） |

## 软件依赖

- Ubuntu 22.04（宿主机）
- Docker + Docker Compose
- VS Code + Dev Containers 扩展（可选）

宿主机需要允许容器访问 USB、网卡和 host 网络（已在 `docker-compose.yml` 和 `.devcontainer/devcontainer.json` 中配置）。

## 快速开始

### 1. 构建并启动容器

```bash
cd /path/to/perception_tower_sensor_env
docker compose -f docker/docker-compose.yml up -d --build
docker exec -it perception_tower bash
```

或在 VS Code 中：
- `Ctrl+Shift+P` → `Dev Containers: Rebuild Container`

容器启动后会自动运行 PTP 同步脚本（见下文“时间同步”）。

### 2. 编译工作区（首次进入容器）

```bash
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

### 3. 启动传感器驱动

**一键启动（推荐）：**

```bash
ros2 launch perception_tower_sensor sensor_env.launch.py
```

指定转台串口：

```bash
ros2 launch perception_tower_sensor sensor_env.launch.py turntable_port:=/dev/ttyUSB1
```

**单独启动（调试时使用）：**

```bash
# 终端 1 - LiDAR
ros2 launch rslidar_sdk start.py

# 终端 2 - 相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端 3 - 转台
ros2 run perception_tower_sensor turntable_node --ros-args --params-file $(ros2 pkg prefix perception_tower_sensor)/share/perception_tower_sensor/config/turntable_params.yaml
```

### 4. 验证话题

```bash
ros2 topic list
ros2 topic hz /fairy/points
ros2 topic hz /camera/color/image_raw
ros2 topic hz /turntable/status
ros2 topic echo /turntable/status
```

## 转台节点

### 协议说明

转台控制节点 `turntable_node` 通过串口与 STM32 通信，支持新版二进制位置帧协议：

- **文本指令**（主机 → 下位机）：`#000P{位置}T{时间}!`、`#000PRST!`、`#000PDST!`、`#000PSTR{间隔ms}!`、`#000PSTP!`
- **二进制位置帧**（下位机 → 主机）：`AA 55 06 01 BATCH POS_H POS_M POS_L DONE CRC8`

节点启动后会自动发送 `#000PSTR{间隔}!` 启动自动位置上报，默认间隔 20 ms（50 Hz），可通过参数 `auto_report_ms` 调整。

### 发布的话题

`/turntable/status`（`perception_tower_sensor_interfaces/TurntableStatus`）

| 字段 | 说明 |
|------|------|
| `header.stamp` | ROS2 系统时间戳（已同步到 LiDAR PTP 时间基准） |
| `header.frame_id` | 固定为 `turntable` |
| `position` | 原始位置值（500 ~ 18000） |
| `angle_deg` | 转换后的角度（°），已应用 `angle_sign` |
| `state` | 转台状态：IDLE / HOMING / MOVING / ERROR |
| `batch` | 当前运动批次号，每次新的自动上报启动后递增 |
| `done` | `true` 表示运动完成/静止，`false` 表示运动中 |

### 可调参数

见 `perception_tower_sensor/config/turntable_params.yaml`：

```yaml
serial_port: /dev/ttyUSB0
serial_baud: 115200
poll_hz: 0.0          # 传统轮询频率；默认 0，使用自动上报
pub_hz: 50.0          # /turntable/status 发布频率
auto_report_ms: 20    # 自动上报间隔，最小 10 ms
pos_origin: 500
deg_per_pos: 0.02
angle_sign: 1
home_timeout_s: 30.0
```

## 时间同步

为了让 LiDAR 点云帧与转台角度在同一时间基准上对齐，系统使用 LiDAR Fairy 作为 PTP 时间源，容器启动后自动同步系统时钟。

### 自动同步（已配置）

容器启动时会通过 `postStartCommand` 自动执行 `docker/start-ptp.sh`：

```bash
ptp4l -i enp4s0 -m -H
phc2sys -s enp4s0 -c CLOCK_REALTIME -m
```

日志保存在容器内 `/var/log/ptp/`。

### 手动验证

```bash
# 检查 PTP 进程
ps aux | grep -E "ptp4l|phc2sys"

# 检查同步日志
tail -f /var/log/ptp/ptp4l.log
tail -f /var/log/ptp/phc2sys.log

# 检查 LiDAR 时间戳类型
grep -E "timestamp_type|ptp" /opt/fairy_ws/config/config.yaml
```

### 修改 LiDAR 网口

如果 LiDAR 不接在 `enp4s0` 上，修改 `docker/docker-compose.yml` 中的环境变量：

```yaml
environment:
  - LIDAR_IFACE=你的网口名
```

或在 `.devcontainer/devcontainer.json` 中增加：

```json
"containerEnv": {
    "LIDAR_IFACE": "你的网口名"
}
```

### 注意事项

- PTP 同步要求 Fairy 开启 PTP 功能。构建镜像时已通过 sed 尝试把 `config.yaml` 中的 `timestamp_type` 设为 `PTP`。
- 如果 `phc2sys` 日志显示未同步，请确认 LiDAR 固件已启用 PTP，或网络中存在 PTP grandmaster。
- 转台节点在收到串口位置帧时用 `rclpy` 的系统时钟打戳；系统时钟经 `phc2sys` 同步后即为 PTP 时间。

## 下游调用

其他机器或项目通过 ROS2 topic 订阅数据即可，无需直接与 LiDAR 做 PTP 同步。

### 订阅示例

```python
import rclpy
from sensor_msgs.msg import PointCloud2
from perception_tower_sensor_interfaces.msg import TurntableStatus

class ConsumerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("consumer_node")
        self.cloud_sub = self.create_subscription(PointCloud2, "/fairy/points", self.on_cloud, 10)
        self.turntable_sub = self.create_subscription(TurntableStatus, "/turntable/status", self.on_turntable, 10)
        self.latest_status = None

    def on_turntable(self, msg):
        self.latest_status = msg

    def on_cloud(self, msg):
        if self.latest_status is None:
            return
        # 按 header.stamp 做时间对齐
        cloud_time = rclpy.time.Time.from_msg(msg.header.stamp)
        table_time = rclpy.time.Time.from_msg(self.latest_status.header.stamp)
        # TODO: 使用历史缓冲做插值，得到 cloud_time 对应的 angle_deg
```

### 时间对齐建议

1. 维护一个 `/turntable/status` 的环形缓冲区（建议保存最近 1~2 秒）。
2. 对每帧/每个点云，按 `header.stamp` 在缓冲区中做线性插值。
3. 用插值得到的 `angle_deg` 对点云做坐标变换。

### 跨机器发现

如果下游机器不在同一台 Ubuntu 上（例如 Mac），参考 `docs/troubleshooting.md` 配置 CycloneDDS 和 Peers。

## 目录结构

```
perception_tower_sensor_env/
├── docker/                      # Docker 构建与 PTP 启动脚本
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start-ptp.sh
├── .devcontainer/               # VS Code Dev Container 配置
│   └── devcontainer.json
├── perception_tower_sensor/     # 转台节点与 launch 文件
│   ├── config/turntable_params.yaml
│   ├── launch/sensor_env.launch.py
│   └── perception_tower_sensor/turntable_node.py
├── perception_tower_sensor_interfaces/  # 消息/服务定义
│   ├── msg/TurntableStatus.msg
│   └── srv/TurntableCommand.srv
├── perception_tower_interfaces/         # 系统级消息/服务定义
│   ├── msg/TurntableStatus.msg
│   └── srv/TurntableCommand.srv
├── docs/                        # 问题排查记录
└── README.md
```

## 常见问题

1. **看不到 `/fairy/points`**：LiDAR topic 名取决于 `rslidar_sdk` 的 `config.yaml` 和 namespace 配置。运行 `ros2 topic list | grep -i fairy` 确认实际 topic 名。
2. **转台无数据**：检查串口权限，`sudo chmod 666 /dev/ttyUSB*` 或把用户加入 `dialout` 组。
3. **PTP 未同步**：检查 `LIDAR_IFACE` 是否正确，以及 LiDAR 是否已启用 PTP master。
