<div align="center">

# SpatialFusion-LM: Foundational Vision Meets SpatialLM

</div>

<div align="justify">

SpatialFusion-LM is a unified framework for spatially grounded 3D scene understanding from monocular or stereo RGB input. It integrates learning-based depth estimation, differentiable point cloud reconstruction, and spatial language modeling into a modular ROS 2 pipeline. By combining geometric cues with linguistic priors, the system generates object-centric 3D layouts that support semantic reasoning, embodied navigation, and robot perception in real-world environments.

The architecture decouples 3D scene inference into three core stages: (1) neural depth prediction, (2) back-projection and point cloud generation, and (3) spatial layout prediction via large-scale language models trained for 3D relational reasoning. Notably, the spatial reasoning is performed over instantaneous point clouds reconstructed in the local camera frame, rather than accumulated global maps, enabling frame-wise layout estimation in dynamic or unstructured environments.

SpatialFusion-LM supports real-time inference, dataset extensibility, and structured logging through Rerun and ROS2, making it suitable for research in vision-language grounding, scene reconstruction, and robotics.

</div>

<p align="center">
  <img src="media/demo_mono_indoor_0.gif" alt="SpatialFusion-LM Mono Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
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
- 🧠 **GPU:** 2x NVIDIA RTX A6000  
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
  <img src="media/demo_stereo_indoor_0.gif" alt="SpatialFusion-LM Stereo Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing stereo depth estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>

### 📸 Monocular+ Demo (RViz Only – Rerun Disabled)

```shell
ros2 launch llm_ros llm_demo.launch.py mode:=mono+ rerun:=false rviz:=true
```

<p align="center">
  <img src="media/demo_mono+_indoor_0_rviz.gif" alt="SpatialFusion-LM Monocular+ Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing stereo monocular+ estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>

## ⚙️ Launch Configuration Options

The `llm_demo.launch.py` file accepts the following arguments:

|   Argument   |  Type   |                               Description                                   |       Default        |
|--------------|---------|-----------------------------------------------------------------------------|----------------------|
| `mode`       | string  | Input mode: `mono`, `mono+`, or `stereo`                                    | `stereo`             |
| `spatialLM`  | bool    | Enable or disable layout prediction via SpatialLM                           | `True`               |
| `rerun`      | bool    | Enable or disable logging to [Rerun](https://rerun.io)                      | `True`               |
| `rviz`       | bool    | Enable or disable RVIZ visualization                                        | `True`               |
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
