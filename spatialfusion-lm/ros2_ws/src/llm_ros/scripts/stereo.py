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
import torch.nn.functional as F
from torch.serialization import add_safe_globals
import numpy.core.multiarray

add_safe_globals([numpy.core.multiarray.scalar])

import os, sys

foundation_dir = os.path.realpath("/FoundationStereo")
sys.path.append(foundation_dir)

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

class FoudationStereoWrapper:
    def __init__(self, node, cv_bridge, baseline):
        self.node = node
        self.cv_bridge = cv_bridge
        self.scale = 0.3
        self.baseline = baseline
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        ckpt_dir = "/FoundationStereo/pretrained_models/model_best_bp2.pth"
        cfg = OmegaConf.load(f'/FoundationStereo/pretrained_models/cfg.yaml')
        args = OmegaConf.create(cfg)
        self.model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()
        self.node.get_logger().info('FoundationStereo model loaded')

    def infer(self, left_image, right_image, left_info, right_info):
        # Resize input images
        img0 = cv2.resize(left_image, fx=self.scale, fy=self.scale, dsize=None)
        img1 = cv2.resize(right_image, fx=self.scale, fy=self.scale, dsize=None)

        H,W = img0.shape[:2]

        img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
        img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

        # Run model
        disp = self.model.forward(img0_tensor, img1_tensor, iters=16, test_mode=True)
        disp = padder.unpad(disp.float())

        disp = disp.data.cpu().numpy().reshape(H,W)

        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf
        K = left_info.copy()
        K[:2] *= self.scale
        depth = K[0,0]*self.baseline/disp
        xyz_map = self.depth2xyzmap(depth, K)
        T = np.array([[0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0]], dtype=np.float64)
        xyz_map = xyz_map @ T.T

        points = xyz_map.reshape(-1, 3)
        colors = img0.reshape(-1, 3) / 255.0

        # Remove NaNs and infs
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        colors = colors[:, [2, 1, 0]]

        timestamp = rclpy.time.Time().to_msg()

        disp_vis = self.vis(depth)
        H_orig, W_orig = left_image.shape[:2]
        disp_vis = cv2.resize(disp_vis, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        return left_image, left_info, right_image, right_info, self.build_depth_image_msg(cv2.resize(depth, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR), timestamp), disp_vis, points, colors, self.build_pointcloud2(points, colors, timestamp)

    def build_pointcloud2(self, points, colors, stamp, frame_id="camera"):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        r = (colors[:, 0] * 255).astype(np.uint8)
        g = (colors[:, 1] * 255).astype(np.uint8)
        b = (colors[:, 2] * 255).astype(np.uint8)
        rgb_float = np.frombuffer(np.stack([b, g, r, np.zeros_like(r)], axis=1).astype(np.uint8).tobytes(), dtype=np.float32)

        cloud_data = np.column_stack((points, rgb_float)).astype(np.float32)

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
    
    def depth2xyzmap(self, depth: np.ndarray, K: np.ndarray, uvs: np.ndarray = None, zmin=0.1):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create meshgrid once
        if uvs is None:
            us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        else:
            uvs = uvs.round().astype(int)
            us, vs = uvs[:, 0], uvs[:, 1]

        # Fast mask and projection
        zs = depth if uvs is None else depth[vs, us]
        invalid_mask = zs < zmin

        xs = (us - cx) * zs / fx if uvs is not None else (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy if uvs is not None else (vs - cy) * zs / fy

        # Build xyz
        xyz = np.stack((xs, ys, zs), axis=-1)
        
        # Build full map
        xyz_map = np.zeros((H, W, 3), dtype=np.float32)
        if uvs is None:
            xyz_map = xyz.reshape(H, W, 3)
        else:
            xyz_map[vs, us] = xyz

        # Invalidate low-depth points
        if np.any(invalid_mask):
            if uvs is None:
                xyz_map[depth < zmin] = 0
            else:
                xyz_map[vs[invalid_mask], us[invalid_mask]] = 0

        return xyz_map