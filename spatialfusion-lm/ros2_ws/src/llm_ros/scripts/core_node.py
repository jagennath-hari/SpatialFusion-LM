#!/usr/bin/env python3

import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, TimeSynchronizer
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
import subprocess

import numpy as np
import cv2

from mono import Unik3DWrapper
from stereo import FoudationStereoWrapper
from rerun_vis import RerunVisualizer
from spatialLM import SpatialLMWrapper

class CoreNode(Node):
    def __init__(self):
        super().__init__('core_node')

        # Declare all expected parameters
        self.declare_parameter("rgb_image", "")
        self.declare_parameter("rgb_info", "")
        self.declare_parameter("left_image", "")
        self.declare_parameter("right_image", "")
        self.declare_parameter("left_info", "")
        self.declare_parameter("right_info", "")
        self.declare_parameter("baseline", 0.0)
        self.declare_parameter("rerun", True)
        self.declare_parameter("spatialLM", True)

        # Get parameter values
        self.rgb_image_topic = self.get_parameter("rgb_image").get_parameter_value().string_value
        self.rgb_info_topic = self.get_parameter("rgb_info").get_parameter_value().string_value
        self.left_image_topic = self.get_parameter("left_image").get_parameter_value().string_value
        self.right_image_topic = self.get_parameter("right_image").get_parameter_value().string_value
        self.left_info_topic = self.get_parameter("left_info").get_parameter_value().string_value
        self.right_info_topic = self.get_parameter("right_info").get_parameter_value().string_value
        self.baseline = self.get_parameter("baseline").get_parameter_value().double_value
        self.use_rerun = self.get_parameter("rerun").get_parameter_value().bool_value
        self.use_llm = self.get_parameter("spatialLM").get_parameter_value().bool_value

        self.cloud_pub = self.create_publisher(PointCloud2, "/spatialLM/cloud", 1)
        self.depth_pub = self.create_publisher(Image, "/spatialLM/depth", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/spatialLM/boxes", 1)
        self.image_overlay_pub = self.create_publisher(Image, "/spatialLM/image", 1)
        self.bridge = CvBridge()

        self.last_infer_time = None
        self.fps_window = deque(maxlen=30)

        # PRIORITY 1: Stereo mode
        if all([self.left_image_topic, self.right_image_topic,
                self.left_info_topic, self.right_info_topic]) and self.baseline > 0.0:
            self.get_logger().info("Stereo mode activated.")
            self.foundation_stereo_wrapper = FoudationStereoWrapper(self, self.bridge, self.baseline)
            if self.use_rerun:
                self.get_logger().info("Rerun enabled. Initializing RerunVisualizer...")
                self.rerun_visualizer = RerunVisualizer("FoundationStereo", spatialLM = self.use_llm)
            else:
                self.rerun_visualizer = None
            if self.use_llm:
                self.spatialLM = SpatialLMWrapper(self, self.bridge)
            else:
                self.spatialLM = None
            self.left_image_sub = Subscriber(self, Image, self.left_image_topic)
            self.right_image_sub = Subscriber(self, Image, self.right_image_topic)
            self.left_info_sub = Subscriber(self, CameraInfo, self.left_info_topic)
            self.right_info_sub = Subscriber(self, CameraInfo, self.right_info_topic)
            self.stereo_sync = TimeSynchronizer(
                [self.left_image_sub, self.right_image_sub, self.left_info_sub, self.right_info_sub], 1)
            self.stereo_sync.registerCallback(self.stereo_callback)

        # PRIORITY 2: Mono + CameraInfo
        elif self.rgb_image_topic and self.rgb_info_topic:
            self.get_logger().info("Mono mode with CameraInfo activated.")
            self.unik3d_wrapper = Unik3DWrapper(self, self.bridge)
            if self.use_rerun:
                self.get_logger().info("Rerun enabled. Initializing RerunVisualizer...")
                self.rerun_visualizer = RerunVisualizer("UniK3D", spatialLM = self.use_llm)
            else:
                self.rerun_visualizer = None
            if self.use_llm:
                self.spatialLM = SpatialLMWrapper(self, self.bridge)
            else:
                self.spatialLM = None
            self.rgb_image_sub = Subscriber(self, Image, self.rgb_image_topic)
            self.rgb_info_sub = Subscriber(self, CameraInfo, self.rgb_info_topic)
            self.rgb_sync = TimeSynchronizer([self.rgb_image_sub, self.rgb_info_sub], 1)
            self.rgb_sync.registerCallback(self.rgb_with_info_callback)

        # PRIORITY 3: Mono only
        elif self.rgb_image_topic:
            self.get_logger().info("Mono mode (image only) activated.")
            self.unik3d_wrapper = Unik3DWrapper(self,self.bridge)
            if self.use_rerun:
                self.get_logger().info("Rerun enabled. Initializing RerunVisualizer...")
                self.rerun_visualizer = RerunVisualizer("UniK3D", spatialLM = self.use_llm)
            else:
                self.rerun_visualizer = None
            if self.use_llm:
                self.spatialLM = SpatialLMWrapper(self, self.bridge)
            else:
                self.spatialLM = None
            self.create_subscription(Image, self.rgb_image_topic, self.rgb_only_callback, 1)

        else:
            self.get_logger().error("Invalid configuration: No valid topic setup detected.")
            self.destroy_node()
            rclpy.shutdown()

    def rgb_only_callback(self, rgb_msg):
        image, intrinsics, depth_msg, depth_vis, points_3d, colors_rgb, cloud_msg = self.unik3d_wrapper.infer(self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8'))
        if self.spatialLM:
            colors_bgr = colors_rgb[:, [2, 1, 0]]
            floor_plan = self.spatialLM.infer(points_3d, colors_bgr)
        else:
            floor_plan = None
        self.depth_pub.publish(depth_msg)
        self.cloud_pub.publish(cloud_msg)
        if self.spatialLM:
            stamp = self.get_clock().now().to_msg()
            self.publish_floorplan_markers(floor_plan, stamp)
            overlay_img = self.project_and_create_overlay(image, intrinsics, floor_plan)
            try:
                self.image_overlay_pub.publish(self.create_img_msg(overlay_img, stamp))
            except TypeError as e:
                pass
        else:
            overlay_img = None
        if self.rerun_visualizer:
            self.rerun_visualizer.log_mono(image, intrinsics, points_3d, colors_rgb, depth_vis, floor_plan, overlay_img)
        #self.log_performance("Mono", check_vram = False)

    def rgb_with_info_callback(self, rgb_msg, info_msg):
        image, intrinsics, depth_msg, depth_vis, points_3d, colors_rgb, cloud_msg = self.unik3d_wrapper.infer(self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8'), intrinsics=np.array(info_msg.k, dtype=np.float64).reshape(3, 3))
        if self.spatialLM:
            colors_bgr = colors_rgb[:, [2, 1, 0]]
            floor_plan = self.spatialLM.infer(points_3d, colors_bgr)
        else:
            floor_plan = None
        self.depth_pub.publish(depth_msg)
        self.cloud_pub.publish(cloud_msg)
        if self.spatialLM:
            stamp = self.get_clock().now().to_msg()
            self.publish_floorplan_markers(floor_plan, stamp)
            overlay_img = self.project_and_create_overlay(image, intrinsics, floor_plan)
            try:
                self.image_overlay_pub.publish(self.create_img_msg(overlay_img, stamp))
            except TypeError as e:
                pass
        else:
            overlay_img = None
        if self.rerun_visualizer:
            self.rerun_visualizer.log_mono(image, intrinsics, points_3d, colors_rgb, depth_vis, floor_plan, overlay_img)
        #self.log_performance("Mono+", check_vram = False)

    def stereo_callback(self, left_img, right_img, left_info, right_info):
        left_image, left_info, right_image, right_info, depth_msg, disp_vis, points_3d, colors_rgb, cloud_msg = self.foundation_stereo_wrapper.infer(self.bridge.imgmsg_to_cv2(left_img, desired_encoding='bgr8'), self.bridge.imgmsg_to_cv2(right_img, desired_encoding='bgr8'), np.array(left_info.k, dtype=np.float64).reshape(3, 3), np.array(right_info.k, dtype=np.float64).reshape(3, 3))
        if self.spatialLM:
            floor_plan = self.spatialLM.infer(points_3d, colors_rgb)
        else:
            floor_plan = None
        self.depth_pub.publish(depth_msg)
        self.cloud_pub.publish(cloud_msg)
        if self.spatialLM:
            stamp = self.get_clock().now().to_msg()
            self.publish_floorplan_markers(floor_plan, stamp)
            overlay_img = self.project_and_create_overlay(left_image, left_info, floor_plan)
            try:
                self.image_overlay_pub.publish(self.create_img_msg(overlay_img, stamp))
            except TypeError as e:
                pass
        else:
            overlay_img = None
        if self.rerun_visualizer:
            self.rerun_visualizer.log_stereo(left_image, right_image, left_info, disp_vis, points_3d, colors_rgb, self.baseline, floor_plan, overlay_img)
        #self.log_performance("Stereo", check_vram = False)


    def publish_floorplan_markers(self, floor_plan, stamp):
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        clear_marker.header.frame_id = "camera"
        clear_marker.header.stamp = stamp
        self.marker_pub.publish(MarkerArray(markers=[clear_marker]))

        marker_array = MarkerArray()
        now = stamp

        for box in floor_plan:
            # Box edges (LINE_LIST)
            marker = Marker()
            marker.header = Header(stamp=now, frame_id="camera")
            marker.ns = box["class"]
            marker.id = box["id"]
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # line width

            # Set color
            if box["class"] == "wall":
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif box["class"] == "door":
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            elif box["class"] == "window":
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            marker.color.a = 1.0

            # Compute 8 corners from center, scale, and rotation
            c = np.array(box["center"])
            s = np.array(box["scale"]) * 0.5  # half-size
            R_mat = np.array(box["rotation"])

            # Corners of unit cube
            corner_offsets = np.array([
                [-1, -1, -1], [1, -1, -1],
                [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1],
                [1, 1, 1], [-1, 1, 1],
            ]) * s

            corners = (R_mat @ corner_offsets.T).T + c

            def pt(xyz):
                return Point(x=xyz[0], y=xyz[1], z=xyz[2])

            # 12 edges for LINE_LIST (each pair is a line)
            edge_indices = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
            ]

            for i0, i1 in edge_indices:
                marker.points.append(pt(corners[i0]))
                marker.points.append(pt(corners[i1]))

            marker_array.markers.append(marker)

            # Floating label above box
            label_marker = Marker()
            label_marker.header = marker.header
            label_marker.ns = f"{box['class']}_label"
            label_marker.id = 10000 + box["id"]  # Unique ID offset
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = c[0]
            label_marker.pose.position.y = c[1]
            label_marker.pose.position.z = c[2] + s[2] + 0.1  # float above top

            label_marker.text = box["label"]
            label_marker.scale.z = 0.2  # font height
            label_marker.color.r = 1.0
            label_marker.color.g = 1.0
            label_marker.color.b = 1.0
            label_marker.color.a = 1.0

            marker_array.markers.append(label_marker)

        self.marker_pub.publish(marker_array)

    def project_and_create_overlay(self, image_bgr, intrinsics, floor_plan):
        if floor_plan is None or len(floor_plan) == 0:
            return

        img = image_bgr.copy()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        height, width = img.shape[:2]

        # This is the same transform used in extract_cloud_data, but inverted
        T = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        T_inv = T.T  # because we did points_3d @ T.T earlier

        for box in floor_plan:
            # Transform center and rotation from Z-up back to camera frame
            c = T_inv @ np.array(box["center"])
            R_mat = T_inv @ np.array(box["rotation"])

            s = np.array(box["scale"]) * 0.5

            # Define corners
            corner_offsets = np.array([
                [-1, -1, -1], [1, -1, -1],
                [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1],
                [1, 1, 1], [-1, 1, 1],
            ]) * s

            corners_3d = (R_mat @ corner_offsets.T).T + c
            pixels = []

            for pt in corners_3d:
                if pt[2] <= 0:
                    pixels.append(None)
                    continue
                u = fx * pt[0] / pt[2] + cx
                v = fy * pt[1] / pt[2] + cy
                pixels.append((int(round(u)), int(round(v))))

            # Draw visible lines
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            color_map = {
                "wall": (0, 0, 255),
                "door": (0, 255, 0),
                "window": (255, 0, 0),
            }
            color = color_map.get(box["class"], (255, 255, 0))

            for i0, i1 in edges:
                pt1 = pixels[i0]
                pt2 = pixels[i1]
                if pt1 is None or pt2 is None:
                    continue
                try:
                    pt1 = (int(pt1[0]), int(pt1[1]))
                    pt2 = (int(pt2[0]), int(pt2[1]))
                    cv2.line(img, pt1, pt2, color, 4, lineType=cv2.LINE_AA)
                except Exception as e:
                    continue

            # Label at first visible corner
            for pt in pixels:
                if pt is not None and 0 <= pt[0] < width and 0 <= pt[1] < height:
                    cv2.putText(img, box["label"], pt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3, cv2.LINE_AA)
                    break

        return img
    
    def create_img_msg(self, img, stamp, frame_id="camera"):
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg
    
    def log_performance(self, tag="", check_vram=False):
        now = time.time()
        if self.last_infer_time is not None:
            dt = now - self.last_infer_time
            self.fps_window.append(1.0 / dt)
            avg_fps = sum(self.fps_window) / len(self.fps_window)

            if check_vram:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits', '-i', '0'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        check=True,
                        text=True
                    )
                    gpu_util, mem_used = result.stdout.strip().split(', ')
                    self.get_logger().info(
                        f"[{tag}] Inference time: {dt*1000:.1f} ms | "
                        f"Avg FPS: {avg_fps:.2f} | GPU: {gpu_util}% | VRAM: {mem_used} MiB"
                    )
                except Exception as e:
                    self.get_logger().warn(f"[{tag}] Failed to query nvidia-smi: {e}")
            else:
                self.get_logger().info(
                    f"[{tag}] Inference time: {dt*1000:.1f} ms | Avg FPS: {avg_fps:.2f}"
                )

        self.last_infer_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CoreNode()
    executor = SingleThreadedExecutor()  # <-- Use single-threaded executor here
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
