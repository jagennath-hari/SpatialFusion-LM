#!/bin/bash

CONTAINER_NAME=ros_llm_dev
USERNAME=$(whoami)
HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🟢 Container '${CONTAINER_NAME}' is already running. Attaching..."
  docker exec -it ${CONTAINER_NAME} bash -c "source /opt/ros/humble/setup.bash && if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi && exec bash"
else
  echo "🚀 Building and launching new container '${CONTAINER_NAME}'..."
  
  docker build --build-arg UID=${HOST_UID} --build-arg GID=${HOST_GID} --build-arg USERNAME=${USERNAME} -t ros_llm ./docker

  xhost +local:docker
  xhost +SI:localuser:$(whoami)

  docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --net=host \
    --pid=host \
    --ipc=host \
    --name ${CONTAINER_NAME} \
    -e DISPLAY=$DISPLAY \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -e O3D_DISABLE_X11_SHM=1 \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/runtime-root:/tmp/runtime-root \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ./spatialfusion-lm/FoundationStereo:/FoundationStereo \
    -v ./spatialfusion-lm/UniK3D:/UniK3D \
    -v ./spatialfusion-lm/SpatialLM:/SpatialLM \
    -v ./datasets:/datasets \
    -v ./spatialfusion-lm/ros2_ws:/ros2_ws \
    --device /dev/dri \
    --user ${HOST_UID}:${HOST_GID} \
    ros_llm
fi
