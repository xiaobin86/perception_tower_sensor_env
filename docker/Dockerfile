# ===== 基础镜像 =====
FROM ros:humble

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ===== 换清华源 =====
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
    -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
    /etc/apt/sources.list && \
    find /etc/apt/sources.list.d -type f -name '*.list' -exec \
        sed -i 's|http://packages.ros.org/ros/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu|g' {} + && \
    apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    python3-pip \
    wget \
    git \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# ===== Orbbec SDK ROS2 驱动依赖 =====
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    python3-rosdep \
    libgflags-dev \
    nlohmann-json3-dev \
    libdw-dev \
    libssl-dev \
    mesa-utils \
    libgl1 \
    libgoogle-glog-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-image-publisher \
    ros-${ROS_DISTRO}-camera-info-manager \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-diagnostic-msgs \
    ros-${ROS_DISTRO}-statistics-msgs \
    ros-${ROS_DISTRO}-backward-ros \
    && rm -rf /var/lib/apt/lists/*

# ===== 安装 CycloneDDS =====
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# ===== 转台节点依赖 =====
RUN apt-get update && apt-get install -y \
    python3-serial \
    && rm -rf /var/lib/apt/lists/*

# ===== 编译 Orbbec 相机驱动 =====
COPY orbbec-v2-main.tar.gz /tmp/orbbec-v2-main.tar.gz
RUN set -e && \
    mkdir -p /opt/orbbec_ws/src/OrbbecSDK_ROS2 && \
    tar -xzf /tmp/orbbec-v2-main.tar.gz -C /opt/orbbec_ws/src/OrbbecSDK_ROS2 --strip-components=1 && \
    rm /tmp/orbbec-v2-main.tar.gz && \
    . /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /opt/orbbec_ws && \
    colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    test -d /opt/orbbec_ws/install/orbbec_camera && \
    # 验证 SDK 版本 - 确保是 v2.8.6（有 USB PAL 支持的版本）
    SDK_FILE=$(find /opt/orbbec_ws/install/orbbec_camera/lib -name "libOrbbecSDK.so.*.*.*" -not -type l | head -1) && \
    echo "SDK file: $SDK_FILE" && \
    strings "$SDK_FILE" | grep -q "LinuxUsbPal" && echo "✓ USB PAL module present" || echo "✗ USB PAL missing" && \
    rm -rf build log src

# ===== RoboSense Fairy LiDAR 驱动依赖 =====
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    libeigen3-dev \
    libboost-dev \
    libyaml-cpp-dev \
    python3-ament-package \
    linuxptp \
    && rm -rf /var/lib/apt/lists/*

# ===== 编译 RoboSense Fairy LiDAR 驱动 =====
COPY rslidar_msg.tar.gz /tmp/rslidar_msg.tar.gz
COPY rslidar_sdk.tar.gz /tmp/rslidar_sdk.tar.gz
RUN set -e && \
    mkdir -p /opt/fairy_ws/src && \
    tar -xzf /tmp/rslidar_msg.tar.gz -C /opt/fairy_ws/src && \
    tar -xzf /tmp/rslidar_sdk.tar.gz -C /opt/fairy_ws/src && \
    rm /tmp/rslidar_msg.tar.gz /tmp/rslidar_sdk.tar.gz && \
    mkdir -p /opt/fairy_ws/config && \
    cp /opt/fairy_ws/src/rslidar_sdk/config/config.yaml /opt/fairy_ws/config/ && \
    . /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /opt/fairy_ws && \
    colcon build --event-handlers console_direct+ --packages-select rslidar_msg rslidar_sdk --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    sed -i "s|config_file = ''|config_file = '/opt/fairy_ws/config/config.yaml'|" /opt/fairy_ws/install/rslidar_sdk/share/rslidar_sdk/launch/start.py && \
    sed -i "s|lidar_type: RSM1|lidar_type: RSFAIRY|" /opt/fairy_ws/config/config.yaml && \
    (grep -q "timestamp_type:" /opt/fairy_ws/config/config.yaml && sed -i "s|timestamp_type:.*|timestamp_type: PTP|" /opt/fairy_ws/config/config.yaml || echo "timestamp_type: PTP" >> /opt/fairy_ws/config/config.yaml) && \
    test -d /opt/fairy_ws/install/rslidar_sdk && \
    rm -rf build log src

# ===== 写入 .bashrc =====
RUN cat >> /root/.bashrc << 'BASHEOF'

# Auto-source ROS2 environment
source /opt/ros/humble/setup.bash 2>/dev/null
source /opt/orbbec_ws/install/setup.bash 2>/dev/null
source /opt/fairy_ws/install/setup.bash 2>/dev/null
[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><Peers><Peer address="192.168.3.187"/></Peers></Discovery><General><Interfaces><NetworkInterface name="wlp2s0"/><NetworkInterface name="enp4s0"/></Interfaces></General></Domain></CycloneDDS>'
BASHEOF
RUN mkdir -p -m 700 /root/.runtime && echo 'export XDG_RUNTIME_DIR=/root/.runtime' >> /root/.bashrc

WORKDIR /workspace
CMD ["bash"]