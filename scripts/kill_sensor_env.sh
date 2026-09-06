#!/bin/bash
# Kill all sensor_env related ROS 2 processes inside the container.
# Usage: ./scripts/kill_sensor_env.sh

set -e

echo "Killing sensor_env processes..."

pkill -9 -f "ros2 launch" || true
pkill -9 -f "turntable_node" || true
pkill -9 -f "rslidar_sdk_node" || true
pkill -9 -f "component_container" || true
pkill -9 -f "tower_node" || true
pkill -9 rviz2 || true
pkill -9 -f "ros2-daemon" || true

echo "Done. Remaining ROS processes:"
ps aux | grep -E "ros2|rviz2|turntable|rslidar|orbbec|tower|camera" | grep -v grep || echo "None"
