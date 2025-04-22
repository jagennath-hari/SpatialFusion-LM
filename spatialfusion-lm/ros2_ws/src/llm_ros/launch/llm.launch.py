import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        # ---------------------------
        # Topic argument declarations
        # ---------------------------
        # Provide the topic name for RGB image (e.g., /camera/image)
        DeclareLaunchArgument('rgb_image', default_value='', description='RGB image topic'),

        # Provide the topic name for RGB CameraInfo (optional)
        DeclareLaunchArgument('rgb_info', default_value='', description='RGB camera info topic'),

        # Provide the topic name for stereo left image
        DeclareLaunchArgument('left_image', default_value='', description='Left stereo image topic'),

        # Provide the topic name for stereo right image
        DeclareLaunchArgument('right_image', default_value='', description='Right stereo image topic'),

        # Provide the topic name for left camera info
        DeclareLaunchArgument('left_info', default_value='', description='Left camera info topic'),

        # Provide the topic name for right camera info
        DeclareLaunchArgument('right_info', default_value='', description='Right camera info topic'),

        # stereo baseline (in meters)
        DeclareLaunchArgument('baseline', default_value='0.0', description='Stereo camera baseline in meters'),

        DeclareLaunchArgument('rerun', default_value='True', description='Use Rerun for visualization'),

        DeclareLaunchArgument('spatialLM', default_value='True', description='Use SpatialLM for layout generation'),

        # ---------------------------
        # Static TF: map -> camera
        # ---------------------------
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_camera_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera'],
            output='screen'
        ),

        # ---------------------------
        # Core Node Launcher
        # ---------------------------
        Node(
            package='llm_ros',
            executable='core_node.py',
            name='core_node',
            output='screen',
            parameters=[{
                'rgb_image': LaunchConfiguration('rgb_image'),
                'rgb_info': LaunchConfiguration('rgb_info'),
                'left_image': LaunchConfiguration('left_image'),
                'right_image': LaunchConfiguration('right_image'),
                'left_info': LaunchConfiguration('left_info'),
                'right_info': LaunchConfiguration('right_info'),
                'baseline': LaunchConfiguration('baseline'),
                'rerun': LaunchConfiguration('rerun'),
                'spatialLM': LaunchConfiguration('spatialLM'),
            }],
        #env={**os.environ, 'CUDA_LAUNCH_BLOCKING': '1'}
        )
    ])
