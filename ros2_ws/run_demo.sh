#!/usr/bin/env bash
# コンテナ内でビルドして launch するためのスクリプト.
#   docker run --rm -it -v "$PWD":/work agripioneer-ros /work/ros2_ws/run_demo.sh [秒数]
# ROS の setup.bash は未定義変数を参照するため -u は付けない
set -eo pipefail
DURATION="${1:-60}"

source /opt/ros/humble/setup.bash
cd /work/ros2_ws
colcon build --symlink-install --packages-select agripioneer_ros
source install/setup.bash

echo "--- ros2 launch agripioneer_ros agripioneer.launch.py (${DURATION}s) ---"
timeout --preserve-status "${DURATION}" \
    ros2 launch agripioneer_ros agripioneer.launch.py || true
