#!/usr/bin/env python3

"""
SPDX-License-Identifier: GPL-3.0-or-later

Copyright (c) 2025 Jagennath Hari

This file is part of SpatialFusion-LM.

SpatialFusion-LM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SpatialFusion-LM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SpatialFusion-LM. If not, see <https://www.gnu.org/licenses/>.
"""

import rclpy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import cv2
import numpy as np
import torch
from unik3d.models import UniK3D
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

class Unik3DWrapper:
    def __init__(self, node, cv_bridge):
        self.node = node
        self.cv_bridge = cv_bridge
        self.model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.node.get_logger().info('UniK3D model loaded')

    def infer(self, image, intrinsics=None):
        rgb_tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(self.device)
        if intrinsics is not None:            
            predictions = self.model.infer(rgb_tensor, self.create_intresics_input(intrinsics))
        else:
            predictions = self.model.infer(rgb_tensor)

        xyz = predictions["points"]
        xyz_np = xyz.squeeze(0).permute(1, 2, 0).cpu().numpy()
        depth_np = predictions["depth"].squeeze().cpu().numpy()

        if intrinsics is None:
            intrinsics = self.estimate_intrinsics(xyz_np, depth_np)

        points_3d, colors_rgb = self.extract_cloud_data(xyz, rgb_tensor)
        timestamp = rclpy.time.Time().to_msg()

        return image, intrinsics, self.build_depth_image_msg(depth_np, timestamp), self.vis(depth_np), points_3d, colors_rgb, self.build_pointcloud2(points_3d, colors_rgb, timestamp)
    
    def create_intresics_input(self, intrinsics):
        params_list = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]] + [0.0] * 12
        params_tensor = torch.tensor(params_list, dtype=torch.float32).to(self.device)
        return OPENCV(params=params_tensor)

    def estimate_intrinsics(self, xyz, depth_map):
        H, W = depth_map.shape
        cx, cy = W / 2, H / 2
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        X = xyz[..., 0].reshape(-1)
        Y = xyz[..., 1].reshape(-1)
        Z = xyz[..., 2].reshape(-1)
        u = u.reshape(-1)
        v = v.reshape(-1)

        valid = np.isfinite(Z) & (Z > 1e-3) & np.isfinite(X) & np.isfinite(Y)
        fx_vals = ((u[valid] - cx) * Z[valid]) / X[valid]
        fy_vals = ((v[valid] - cy) * Z[valid]) / Y[valid]

        fx = np.median(fx_vals)
        fy = np.median(fy_vals)

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float64)
        return K
    
    def extract_cloud_data(self, xyz, rgb_tensor, stride=4):
        xyz = xyz[:, :, ::stride, ::stride]
        rgb_tensor = rgb_tensor[:, ::stride, ::stride]

        points_3d = xyz.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        T = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points_3d = points_3d @ T.T
        colors_rgb = rgb_tensor.permute(1, 2, 0).reshape(-1, 3).cpu().numpy() / 255.0

        valid_mask = np.isfinite(points_3d).all(axis=1)
        return points_3d[valid_mask], colors_rgb[valid_mask]

    def build_pointcloud2(self, points_3d, colors_rgb, stamp, frame_id="camera"):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        r = (colors_rgb[:, 0] * 255).astype(np.uint8)
        g = (colors_rgb[:, 1] * 255).astype(np.uint8)
        b = (colors_rgb[:, 2] * 255).astype(np.uint8)
        rgb_float = np.frombuffer(np.stack([b, g, r, np.zeros_like(r)], axis=1).astype(np.uint8).tobytes(), dtype=np.float32)

        cloud_data = np.column_stack((points_3d, rgb_float)).astype(np.float32)

        return PointCloud2(
            header=header,
            height=1,
            width=len(cloud_data),
            is_dense=False,
            is_bigendian=False,
            fields=[
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ],
            point_step=16,
            row_step=16 * len(cloud_data),
            data=cloud_data.tobytes()
        )
    
    def vis(self, disp, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO):
        disp = disp.copy()
        depth_map_normalized = cv2.equalizeHist(cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
        return cv2.applyColorMap(depth_map_normalized, color_map)[..., ::-1].astype(np.uint8)
    
    def build_depth_image_msg(self, depth_np, stamp, frame_id="map"):
        msg = self.cv_bridge.cv2_to_imgmsg(depth_np.astype(np.float32), encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg