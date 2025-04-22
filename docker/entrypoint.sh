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

# Ensure editable install of UniK3D every time container starts
if [ -d "/UniK3D" ]; then
    pip3 install -e /UniK3D 
fi

# Pre-download UniK3D model
echo "📦 Checking UniK3D model..."
python3 -c "from unik3d.models import UniK3D; UniK3D.from_pretrained('lpiccinelli/unik3d-vitl')" || {
  echo "❌ Failed to download UniK3D model. Please check internet connection or model ID."
}

exec "$@"
