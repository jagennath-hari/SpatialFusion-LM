FROM nvidia/cuda:12.8.0-devel-ubuntu22.04


ARG ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Step 1: Install base dependencies
RUN apt update && apt install -y \
    curl gnupg2 lsb-release locales software-properties-common \
    tzdata && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    add-apt-repository universe

# Step 2: Add ROS 2 GPG key and source list
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# Step 3: Install ROS and colcon tools
RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-desktop \
    python3-pip \
    python3-colcon-common-extensions \
    build-essential git \
    ros-dev-tools \
    libsparsehash-dev \
    libgtk-3-dev \
    libxkbcommon-x11-0 \
    vulkan-tools

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 scipy && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir tokenizers poethepoet transformers \
    scikit-image omegaconf opencv-contrib-python imgaug ninja timm albumentations jupyterlab shapely terminaltables \
    joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d transformations einops rerun-sdk bbox addict toml \
    gdown flash-attn nodejs tqdm wandb termcolor tabulate tables pandas h5py setuptools safetensors pandas huggingface_hub && \
    pip3 install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 uninstall torchsparse && \
    pip3 install --upgrade pip
    
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"
RUN FORCE_CUDA=1 pip3 install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git
    

RUN apt update && apt install -y \
    libpcl-dev \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-msgs \
    ros-${ROS_DISTRO}-pcl-ros \
    libflann-dev

# Copy UniK3D and install it
COPY ./UniK3D /UniK3D
RUN pip3 install -e /UniK3D

ARG USERNAME=rosuser
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME
RUN python3 -c "from unik3d.models import UniK3D; UniK3D.from_pretrained('lpiccinelli/unik3d-vitl')"
USER root
    
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]