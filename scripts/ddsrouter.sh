#!/bin/bash
# DDS Router launcher for perception-tower
# Usage: ddsrouter.sh <config_file.yaml>
#
# Example config files are in: /mnt/d/work/perception-tower/third_party/dds-router/ddsrouter_yaml/

DDSROUTER_DIR="/mnt/d/work/perception-tower/third_party/dds-router"
export LD_LIBRARY_PATH="${DDSROUTER_DIR}/fastdds/lib:${DDSROUTER_DIR}/fastcdr/lib:${DDSROUTER_DIR}/ddsrouter_core/lib:${DDSROUTER_DIR}/ddspipe_core/lib:${DDSROUTER_DIR}/ddspipe_participants/lib:${DDSROUTER_DIR}/ddspipe_yaml/lib:${DDSROUTER_DIR}/ddsrouter_yaml/lib:${DDSROUTER_DIR}/ddsrouter_tool/lib:${DDSROUTER_DIR}/cpp_utils/lib:${DDSROUTER_DIR}/foonathan_memory_vendor/lib:$LD_LIBRARY_PATH"

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file.yaml>"
    echo ""
    echo "Example:"
    echo "  $0 my_config.yaml"
    exit 1
fi

exec "${DDSROUTER_DIR}/ddsrouter_tool/bin/ddsrouter" -c "$1" "${@:2}"
