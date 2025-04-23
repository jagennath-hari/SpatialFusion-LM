<div align="center">

# SpatialFusion-LM: Foundational Vision Meets SpatialLM

</div>

<div align="justify">

SpatialFusion-LM is a unified framework for spatially grounded 3D scene understanding from monocular or stereo RGB input. It integrates learning-based depth estimation, differentiable point cloud reconstruction, and spatial language modeling into a modular ROS 2 pipeline. By combining geometric cues with linguistic priors, the system generates object-centric 3D layouts that support semantic reasoning, embodied navigation, and robot perception in real-world environments.

The architecture decouples 3D scene inference into three core stages: (1) neural depth prediction, (2) back-projection and point cloud generation, and (3) spatial layout prediction via large-scale language models trained for 3D relational reasoning. Notably, the spatial reasoning is performed over instantaneous point clouds reconstructed in the local camera frame, rather than accumulated global maps, enabling frame-wise layout estimation in dynamic or unstructured environments.

SpatialFusion-LM supports real-time inference, dataset extensibility, and structured logging through Rerun and ROS2, making it suitable for research in vision-language grounding, scene reconstruction, and robotics.

</div>

<p align="center">
  <img src="media/demo_mono_indoor_0.gif" alt="SpatialFusion-LM Monocular LLM Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing monocular depth estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>

## 🔧 Features

- 📷 Supports monocular, monocular+ and stereo vision
- 🔍 Neural depth estimation with metric 3D reconstruction
- 🧱 Differentiable point cloud generation in the camera frame
- 🧠 Language-conditioned spatial layout prediction
- 🧩 Modular ROS2 architecture (plug-and-play components)
- 🌀 Real-time inference and visualization
- 📊 Integrated logging via [Rerun](https://www.rerun.io/)

## ⚙️ Setup
### 🖥️ Tested Configuration
SpatialFusion-LM has been tested on:

- 🐧 **Ubuntu:** 24.04  
- 🧠 **GPU:** NVIDIA RTX A6000  
- ⚙️ **CUDA:** 12.8  
- 🧊 **Environment:** Docker container with GPU support

> Other modern Ubuntu + CUDA setups may work, but this is the validated reference configuration.

### 🧬 Clone the repository and its submodules

```shell
git clone --recursive https://github.com/jagennath-hari/SpatialFusion-LM.git && cd SpatialFusion-LM
```

### 📥 Download the model (FoundationStereo and SpatialLM) weights

```shell
bash scripts/download_weights.sh
```

### 🗃️ Download datasets

This script will prompt you to select one or more datasets to download:

```shell
bash scripts/download_dataset.sh
```

## 🐳 Run in Docker

The easiest way to launch SpatialFusion-LM is via Docker. The following command will automatically build the image (if needed) and run the container with full GPU and ROS2 support:

```shell
bash run_container.sh
```

## 🧪 Demo

```shell
ros2 launch llm_ros llm_demo.launch.py
```

<p align="center">
  <img src="media/demo_stereo_indoor_0.gif" alt="SpatialFusion-LM Stereo LLM Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing stereo depth estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>

### 📷 Monocular+ Demo (RViz Only – Rerun Disabled)

```shell
ros2 launch llm_ros llm_demo.launch.py mode:=mono+ rerun:=false rviz:=true
```

<p align="center">
  <img src="media/demo_mono+_indoor_0_rviz.gif" alt="SpatialFusion-LM Monocular+ LLM Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing stereo monocular+ estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>

## ⚙️ Launch Configuration Options

The `llm_demo.launch.py` file accepts the following arguments:

|   Argument   |  Type   |                               Description                                   |       Default        |
|:-------------|:-------:|:---------------------------------------------------------------------------:|---------------------:|
| `mode`       | string  | Input mode: `mono`, `mono+`, or `stereo`                                    | `stereo`             |
| `spatialLM`  | bool    | Enable or disable layout prediction via SpatialLM                           | `true`               |
| `rerun`      | bool    | Enable or disable logging to [Rerun](https://rerun.io)                      | `true`               |
| `rviz`       | bool    | Enable or disable RVIZ visualization                                        | `true`               |
| `bag_path`   | string  | Path to the ROS2 bag file (e.g., `/datasets/indoor_0`)                      | `/datasets/indoor_0` |

### 📸 Mono, 📷 Mono+, 📷 📷 Stereo?

<p align="center">
<pre>
                                +---------------------------------+
                                |              mode=?             |
                                +---------------------------------+
                                                |
                                  ┌─────────────┴──────────────┐
                                  │             │              │
                                 mono          mono+        stereo
                                  │             │              │
                                 rgb           rgb          camera
                                               intr.         intr.
                                                +              +
                                               rgb           stereo
                                                              pair
                                                               +
                                                            baseline
</pre>
</p>

#### 🤖 Mode Descriptions

- **`mono`** – Only an RGB image is provided  
  [**UniK3D**](https://lpiccinelli-eth.github.io/pub/unik3d/) internally estimates camera intrinsics and uses them to predict **metric (absolute) depth**.  
  While this enables 3D reconstruction without calibration, the accuracy depends on the quality of intrinsic estimation.  
  🚀 Suitable for quick deployment or uncalibrated cameras.

- **`mono+`** – RGB image **and** accurate camera intrinsics are provided  
  [**UniK3D**](https://lpiccinelli-eth.github.io/pub/unik3d/) uses the supplied intrinsics to produce **more accurate metric depth**, with better scale alignment.  
  🧪 Ideal for calibrated cameras (e.g., using `/camera_info`).

- **`stereo`** – Left and right **rectified images**, intrinsics, and baseline are required  
  [**FoundationStereo**](https://nvlabs.github.io/FoundationStereo/) performs dense stereo matching to compute **high-precision, metric depth** via triangulation.  
  🛡️ This mode is the most **robust and accurate**, especially in real-world or textured environments.

## 🖼️ Demo Gallery

Below are example configurations showing how SpatialFusion-LM behaves with different launch options. 

---

### 📸 Mono 🧠 SpatialLM Disabled (mono, rerun)

```shell
ros2 launch llm_ros llm_demo.launch.py mode:=mono spatialLM:=false rerun:=true rviz:=false
```

<p align="center">
  <img src="media/demo_mono_depth_indoor_0.gif" alt="SpatialFusion-LM Monocular Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing monocular estimation and 3D reconstruction on indoor scene <code>indoor_0</code>.</em>
</p>

### 📷 📷 Stereo 🧠 SpatialLM Disabled (mono, rerun)

```shell
ros2 launch llm_ros llm_demo.launch.py mode:=mono spatialLM:=false rerun:=true rviz:=false
```

<p align="center">
  <img src="media/demo_stereo_depth_indoor.gif" alt="SpatialFusion-LM Stereo Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing stereo estimation and 3D reconstruction on indoor scene <code>indoor_0</code>.</em>
</p>

## 🛠️ Using SpatialFusion-LM with Your Own ROS2 Topics

To run SpatialFusion-LM on a live ROS2 system or your own dataset:

### 1️⃣ Use `llm.launch.py` for direct topic-level control

This version of the launch file allows you to specify **raw topic names directly** (no bag playback or auto setup). Examples:

#### This simulates mode:=mono as there no rgb_info provided.
```shell
ros2 launch llm_ros llm.launch.py \
  rgb_image:=/your_camera/image_rect \
  rerun:=true \
  spatialLM:=true
```

#### This simulates mode:=mono+ as rgb_info provided.
```shell
ros2 launch llm_ros llm.launch.py \
  rgb_image:=/your_camera/image_rect \
  rgb_info:=/your_camera/camera_info \
  rerun:=true \
  spatialLM:=true
```

#### This simulates mode:=stereo as there left and right topics, left_info and right_info, and baseline provided.
```shell
ros2 launch llm_ros llm.launch.py \
  left_image:=/stereo/left/image_rect \
  right_image:=/stereo/right/image_rect \
  left_info:=/stereo/left/camera_info \
  right_info:=/stereo/right/camera_info \
  baseline:=0.12 \
  rerun:=true \
  spatialLM:=true
```

### 2️⃣ Parameter Descriptions
```shell
ros2 launch llm_ros llm.launch.py -s
```
|Parameter | Description | Default | ROS Msg Type |
|:----------|:-----------:|:-------:|--------------:|
|`rgb_image` | RGB image topic | '' | `sensor_msgs/msg/Image` |
|`rgb_info` | RGB camera info topic | '' | `sensor_msgs/msg/CameraInfo` |
|`left_image` | Left stereo image topic | '' | `sensor_msgs/msg/Image` |
|`right_image` | Right stereo image topic | '' | `sensor_msgs/msg/Image` |
|`left_info` | Left camera info topic | '' | `sensor_msgs/msg/CameraInfo` |
|`right_info` | Right camera info topic | '' | `sensor_msgs/msg/CameraInfo` |
|`baseline` | Stereo camera baseline (in meters) | `0.0` | `float` (launch param) |
|`rerun` | Enable Rerun logging | `true` | `bool` (launch param) |
|`spatialLM` | Enable 3D layout prediction via SpatialLM | `true` | `bool` (launch param) |


### 📤 Output Topics

These are the outputs published by the core_node.py:

|Topic|Description|ROS Msg Type|
|:----|:---------:|-----------:|
|`/spatialLM/depth`|	Predicted depth map (1-channel float)|	`sensor_msgs/msg/Image`|
|`/spatialLM/cloud`|	Reconstructed 3D point cloud|	`sensor_msgs/msg/PointCloud2`|
|`/spatialLM/image`|	RGB image with projected 3D layout|	`sensor_msgs/msg/Image`|
|`/spatialLM/boxes`|	Predicted 3D layout objects (e.g., boxes)|	`visualization_msgs/msg/MarkerArray`|
|`/tf`|	Transform tree| (e.g., map → camera)|	`tf2_msgs/msg/TFMessage`|