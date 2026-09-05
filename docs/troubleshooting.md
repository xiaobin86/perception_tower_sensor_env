# Perception Tower 问题排查记录

## 1. LiDAR ERRCODE_MSOPTIMEOUT

**现象**: rslidar_sdk 启动后持续报 `ERRCODE_MSOPTIMEOUT`，收不到 LiDAR 数据。

**原因**:
1. config.yaml 中 `lidar_type` 配置错误（默认是 RSM1，应改为 RSFAIRY）
2. devcontainer.json 没有设置 `network_mode: host`，容器用的是桥接网络，UDP 数据包进不来

**解决**:
- 修改 config.yaml: `lidar_type: RSFAIRY`
- devcontainer.json 添加:
  ```json
  "runArgs": ["--network=host", "--privileged", "--device=/dev"]
  ```

---

## 2. LiDAR config.yaml 路径问题

**现象**: `The format of config file /opt/fairy_ws/src/rslidar_sdk/config/config.yaml is wrong`

**原因**: launch 文件中 `config_file = ''` 为空，默认找 `src` 下的路径，但编译后 `src` 已被删除。

**解决**:
- 编译后将 config 复制到 `/opt/fairy_ws/config/`
- 修改 launch 文件: `config_file = '/opt/fairy_ws/config/config.yaml'`
- Dockerfile 中用 sed 持久化修改

---

## 3. LiDAR IP 发现

**现象**: 不知道 LiDAR 的 IP 地址。

**解决**:
- LiDAR 通过网线连接到 `enp4s0` 接口
- LiDAR IP: `192.168.1.200`
- 主机 `enp4s0` 需要配置同网段 IP（如 `192.168.1.102`）

---

## 4. Orbbec USB 后端不可用

**现象**: `USB backend is unavailable; continuing with network device enumeration`

**原因**: 缺少 USB 相关依赖包。

**解决**: Dockerfile 中添加:
```dockerfile
libusb-1.0-0-dev
libudev-dev
usbutils
```

---

## 5. XDG_RUNTIME_DIR 权限问题

**现象**: `QStandardPaths: wrong permissions on runtime directory /tmp/runtime-root, 0755 instead of 0700`

**解决**:
- 改到 `/root/.runtime`（避免被 /tmp 清理）
- 设置权限 0700:
  ```dockerfile
  mkdir -p -m 700 /root/.runtime
  export XDG_RUNTIME_DIR=/root/.runtime
  ```

---

## 6. Mac 看不到 Ubuntu 的 ROS2 Topic

**现象**: Mac 上 `ros2 topic list` 为空，Ubuntu 容器里的 topic 看不到。

**原因**: ROS2 Humble 默认用 FastDDS，跨机器需要配置 DDS 走正确的网络接口。

**网络拓扑**:
- WiFi (wlp2s0): 192.168.3.162 ←→ Mac: 192.168.3.187
- 有线 (enp4s0): 192.168.1.102 ←→ LiDAR: 192.168.1.200

**解决**:

Ubuntu 容器内安装 CycloneDDS:
```bash
apt-get install -y ros-humble-rmw-cyclonedds-cpp
```

Ubuntu 容器内设置环境变量:
```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><Peers><Peer address="192.168.3.187"/></Peers></Discovery><General><Interfaces><NetworkInterface name="wlp2s0"/><NetworkInterface name="enp4s0"/></Interfaces></General></Domain></CycloneDDS>'
```

Mac 上:
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><Peers><Peer address="192.168.3.162"/></Peers></Discovery><General><Interfaces><NetworkInterface name="en0"/></Interfaces></General></Domain></CycloneDDS>'
ros2 topic list
```

---

## 7. Docker Build 中 sed 在 colcon build 之前执行

**现象**: `sed: can't read /opt/fairy_ws/install/rslidar_sdk/share/rslidar_sdk/launch/start.py: No such file or directory`

**原因**: sed 修改 install 目录下的文件，但 install 目录是 colcon build 之后才创建的。

**解决**: 调整 Dockerfile 中命令顺序，sed 放在 colcon build 之后。

---

## 8. conda 环境兼容性问题（已弃用）

**现象**: conda 的 GCC 15.3 交叉编译器导致严重 ABI 不兼容，backward_ros 符号链接失败、uint32_t 错误等。

**解决**: 放弃 conda 方案，改用 Docker + `ros:humble` 基础镜像（apt 安装 ROS2）。

---

## 网络拓扑总结

```
Mac (192.168.3.187)
    ↓ WiFi
Ubuntu wlp2s0 (192.168.3.162)
    ↓
Ubuntu enp4s0 (192.168.1.102)
    ↓ 网线
LiDAR (192.168.1.200)
```
