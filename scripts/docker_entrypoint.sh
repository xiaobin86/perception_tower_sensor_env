#!/bin/bash
set -e

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f /opt/orbbec_ws/install/setup.bash ]; then
    source /opt/orbbec_ws/install/setup.bash
fi

if [ -f /opt/fairy_ws/install/setup.bash ]; then
    source /opt/fairy_ws/install/setup.bash
fi

if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"