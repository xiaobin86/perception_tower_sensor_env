#!/bin/bash
# Start PTP synchronization for LiDAR time base.
# This script is invoked by devcontainer.json postStartCommand.

set -e

LIDAR_IFACE=${LIDAR_IFACE:-enp4s0}

if ! command -v ptp4l >/dev/null 2>&1; then
    echo "[ptp] ptp4l not found; skipping PTP sync"
    exit 0
fi

if [ ! -d "/sys/class/net/$LIDAR_IFACE" ]; then
    echo "[ptp] network interface $LIDAR_IFACE not found; skipping PTP sync"
    exit 0
fi

# Stop any existing instances to avoid duplicates after container restart.
pkill -x ptp4l || true
pkill -x phc2sys || true
sleep 1

mkdir -p /var/log/ptp
echo "[ptp] starting ptp4l on $LIDAR_IFACE"
ptp4l -i "$LIDAR_IFACE" -m -H -l 5 >/var/log/ptp/ptp4l.log 2>&1 &

sleep 2

echo "[ptp] starting phc2sys"
phc2sys -s "$LIDAR_IFACE" -c CLOCK_REALTIME -m -l 5 >/var/log/ptp/phc2sys.log 2>&1 &

echo "[ptp] PTP sync started on $LIDAR_IFACE"
