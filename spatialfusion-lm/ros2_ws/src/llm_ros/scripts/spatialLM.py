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

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

import os
import sys

spatiallm_dir = os.path.realpath("/SpatialLM")
sys.path.append(spatiallm_dir)

from spatiallm import Layout
from spatiallm import SpatialLMLlamaForCausalLM, SpatialLMQwenForCausalLM
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose

class SpatialLMWrapper:
    def __init__(self, node, cv_bridge):
        self.node = node
        self.cv_bridge = cv_bridge
        # load the model
        self.model_path = "/SpatialLM/manycore-research/SpatialLM-Llama-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to("cuda")
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()

    def infer(self, points_3d, colors_rgb):
        grid_size = Layout.get_grid_size()
        num_bins = Layout.get_num_bins()
        input_pcd = self.preprocess_point_cloud(points_3d, colors_rgb, grid_size, num_bins)
        
        try:
            # generate the layout
            layout = self.generate_layout(
                self.model,
                input_pcd,
                self.tokenizer,
                "/SpatialLM/code_template.txt",
                10,
                0.95,
                0.6,
                1,
            )
            layout.translate(np.min(points_3d, axis=0))
            floor_plan = layout.to_boxes()

        except RuntimeError as e:
            self.node.get_logger().warn(f"⚠️ SpatialLM generation failed: {e}")
            floor_plan = []

        return floor_plan

    def preprocess_point_cloud(self, points, colors, grid_size, num_bins):
        transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="test",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=num_bins,
                ),
            ]
        )
        point_cloud = transform(
            {
                "name": "pcd",
                "coord": points.copy(),
                "color": colors.copy(),
            }
        )
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        rgb = point_cloud["color"]
        point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
        return torch.as_tensor(np.stack([point_cloud], axis=0))
    
    def generate_layout(
        self,
        model,
        point_cloud,
        tokenizer,
        code_template_file,
        top_k=2,
        top_p=0.98,
        temperature=0.6,
        num_beams=1,
        max_new_tokens=4096,
    ):
        # load the code template
        with open(code_template_file, "r") as f:
            code_template = f.read()

        prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"

        # prepare the conversation data
        if model.config.model_type == SpatialLMLlamaForCausalLM.config_class.model_type:
            conversation = [{"role": "user", "content": prompt}]
        elif model.config.model_type == SpatialLMQwenForCausalLM.config_class.model_type:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")

        input_ids = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            {"input_ids": input_ids, "point_clouds": point_cloud},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )
        model.generate(**generate_kwargs)

        generate_texts = []
        for text in streamer:
            generate_texts.append(text)

        layout_str = "".join(generate_texts)
        layout = Layout(layout_str)
        layout.undiscretize_and_unnormalize()
        
        
        return layout