#!/usr/bin/env python3

import numpy as np
import cv2
import rerun as rr
from scipy.spatial.transform import Rotation as R

class RerunVisualizer:
    def __init__(self, name, spatialLM=True):
        final_name = f"Spatial{name}LLM" if spatialLM else name
        rr.init(final_name, spawn=True)

    def log_mono(self, image_bgr, intrinsics, points_3d, colors_rgb, depth_map, floor_plan = None, overlay_bgr = None, frame = "camera"):
        rr.log(f"{frame}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        T = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        quat = R.from_matrix(T).as_quat()

        rr.log(f"{frame}/rgb", rr.Transform3D(
            translation=[0.0, 0.0, 0.0],
            rotation=rr.Quaternion(xyzw=quat.tolist())
        ))

        rr.log(f"{frame}/rgb/image", rr.Pinhole(
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]],
        ))

        rr.log(f"{frame}/rgb/image", rr.Image(image_rgb))

        rr.log(f"{frame}/rgb/depth", rr.Image(depth_map))

        rr.log(f"{frame}/cloud", rr.Points3D(positions=points_3d, colors=colors_rgb))

        if floor_plan is not None:
            rr.log(f"{frame}/pred", rr.Clear(recursive=True))
            for box in floor_plan:
                uid = box["id"]
                group = box["class"]
                label = box["label"]

                rr.log(
                    f"{frame}/pred/{group}/{uid}",
                    rr.Boxes3D(
                        centers=box["center"],
                        half_sizes=0.5 * box["scale"],
                        labels=label,
                    ),
                    rr.InstancePoses3D(mat3x3=box["rotation"]),
                    static=False,
                )

        # Overlay image
        if overlay_bgr is not None:
            image_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"{frame}/rgb/overlay", rr.Image(image_overlay_rgb))

    def log_stereo(self, left_img, right_img, intrinsics, disparity_img, points, colors, baseline = 0.0, floor_plan = None, overlay_bgr = None, frame = "camera"):
        rr.log(f"{frame}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Convert to RGB for logging
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Rotation matrix to convert to Rerun's expected frame
        T_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        quat = R.from_matrix(T_cam).as_quat()  # x,y,z,w

        # Log left camera (origin)
        rr.log(f"{frame}/left", rr.Transform3D(
            translation=[0.0, 0.0, 0.0],
            rotation=rr.Quaternion(xyzw=quat.tolist())
        ))
        rr.log(f"{frame}/left/image", rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ))
        rr.log(f"{frame}/left/image", rr.Image(left_rgb))
        rr.log(f"{frame}/left/disparity", rr.Image(disparity_img))

        # Compute rotated baseline offset
        baseline_offset = T_cam @ np.array([baseline, 0.0, 0.0])

        # Log right camera
        rr.log(f"{frame}/right", rr.Transform3D(
            translation=(baseline_offset).tolist(),
            rotation=rr.Quaternion(xyzw=quat.tolist())
        ))
        rr.log(f"{frame}/right/image", rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ))
        rr.log(f"{frame}/right/image", rr.Image(right_rgb))

        # Log point cloud
        rr.log(f"{frame}/cloud", rr.Points3D(positions=points, colors=colors))

        if floor_plan is not None:
            rr.log(f"{frame}/pred", rr.Clear(recursive=True))
            for box in floor_plan:
                uid = box["id"]
                group = box["class"]
                label = box["label"]

                rr.log(
                    f"{frame}/pred/{group}/{uid}",
                    rr.Boxes3D(
                        centers=box["center"],
                        half_sizes=0.5 * box["scale"],
                        labels=label,
                    ),
                    rr.InstancePoses3D(mat3x3=box["rotation"]),
                    static=False,
                )

        # Overlay image
        if overlay_bgr is not None:
            image_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"{frame}/left/overlay", rr.Image(image_overlay_rgb))
        