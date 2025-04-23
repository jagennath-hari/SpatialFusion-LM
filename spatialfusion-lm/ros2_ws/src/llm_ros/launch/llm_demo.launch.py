import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction, TimerAction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition


def setup_mode_and_bag(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)
    spatialLM_enabled = LaunchConfiguration('spatialLM').perform(context).lower() == 'true'
    bag_path = LaunchConfiguration('bag_path').perform(context)
    rate = '0.5' if spatialLM_enabled else '1.0'

    topic_map = {
        'stereo': {
            'left_image': '/zed/zed_node/left/image_rect_color',
            'right_image': '/zed/zed_node/right/image_rect_color',
            'left_info': '/zed/zed_node/left/camera_info',
            'right_info': '/zed/zed_node/right/camera_info',
        },
        'mono+': {
            'rgb_image': '/zed/zed_node/rgb/image_rect_color',
            'rgb_info': '/zed/zed_node/rgb/camera_info',
        },
        'mono': {
            'rgb_image': '/zed/zed_node/rgb/image_rect_color',
        },
    }

    config_setters = []
    for key, value in topic_map.get(mode, {}).items():
        config_setters.append(SetLaunchConfiguration(key, value))

    config_setters.append(
        TimerAction(
            period=10.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'bag', 'play', bag_path, '--rate', rate],
                    output='screen'
                )
            ]
        )
    )

    return config_setters


def generate_launch_description():
    default_bag_path = '/dataset/indoor_0'
    default_rviz_config = '/ros2_ws/src/llm_ros/config/llm.rviz'

    return LaunchDescription([
        # Mode and flags
        DeclareLaunchArgument('mode', default_value='stereo'),
        DeclareLaunchArgument('spatialLM', default_value='True'),
        DeclareLaunchArgument('rerun', default_value='True'),
        DeclareLaunchArgument('rviz', default_value='True'),
        DeclareLaunchArgument('bag_path', default_value=default_bag_path),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),

        # Topics
        DeclareLaunchArgument('rgb_image', default_value=''),
        DeclareLaunchArgument('rgb_info', default_value=''),
        DeclareLaunchArgument('left_image', default_value=''),
        DeclareLaunchArgument('right_image', default_value=''),
        DeclareLaunchArgument('left_info', default_value=''),
        DeclareLaunchArgument('right_info', default_value=''),
        DeclareLaunchArgument('baseline', default_value='0.12'),

        # Launch bag + topic config
        OpaqueFunction(function=setup_mode_and_bag),

        # TF: map → camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_camera_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera'],
            output='screen'
        ),

        # Core node
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
        ),

        # Conditionally launch RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
