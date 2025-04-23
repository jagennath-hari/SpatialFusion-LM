<div align="center">

# SpatialFusion-LM: Foundational Vision Meets SpatialLM

</div>

<div align="justify">

SpatialFusion-LM is a unified framework for spatially grounded 3D scene understanding from monocular or stereo RGB input. It integrates learning-based depth estimation, differentiable point cloud reconstruction, and spatial language modeling into a modular ROS 2 pipeline. By combining geometric cues with linguistic priors, the system generates object-centric 3D layouts that support semantic reasoning, embodied navigation, and robot perception in real-world environments.

The architecture decouples 3D scene inference into three core stages: (1) neural depth prediction, (2) back-projection and point cloud generation, and (3) spatial layout prediction via large-scale language models trained for 3D relational reasoning. Notably, the spatial reasoning is performed over instantaneous point clouds reconstructed in the local camera frame, rather than accumulated global maps, enabling frame-wise layout estimation in dynamic or unstructured environments.

SpatialFusion-LM supports real-time inference, dataset extensibility, and structured logging through Rerun and ROS2, making it suitable for research in vision-language grounding, scene reconstruction, and robotics.

</div>

<p align="center">
  <img src="media/demo_mono_indoor_0.gif" alt="SpatialFusion-LM Indoor Demo" style="max-width: 100%; height: auto;"/><br/>
  <em>SpatialFusion-LM performing monocular depth estimation, 3D reconstruction, and spatial layout prediction on indoor scene <code>indoor_0</code>.</em>
</p>
