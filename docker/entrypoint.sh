#!/bin/bash
set -e

ROS_DISTRO=humble

# Source ROS environment
source /opt/ros/${ROS_DISTRO}/setup.bash

# If workspace not built, build it
if [ ! -f /ros2_ws/install/setup.bash ]; then
  echo "🔧 ROS 2 workspace not built. Running colcon build..."
  cd /ros2_ws
  chmod +x /ros2_ws/src/llm_ros/scripts/core_node.py
  colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) # build the workspace
fi

# Source your ROS 2 workspace
source /ros2_ws/install/setup.bash

exec "$@"
