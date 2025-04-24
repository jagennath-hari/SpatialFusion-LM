import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction, TimerAction
)
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition


def setup_tum_topics_and_bag(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)
    spatialLM_enabled = LaunchConfiguration('spatialLM').perform(context).lower() == 'true'
    bag_path = LaunchConfiguration('bag_path').perform(context)
    rate = '0.5' if spatialLM_enabled else '1.0'

    topic_map = {
        'mono+': {
            'rgb_image': '/rgb/image',
            'rgb_info': '/rgb/image_info',
        },
        'mono': {
            'rgb_image': '/rgb/image',
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
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('mode', default_value='mono+'),
        DeclareLaunchArgument('spatialLM', default_value='True'),
        DeclareLaunchArgument('rerun', default_value='True'),
        DeclareLaunchArgument('rviz', default_value='True'),
        DeclareLaunchArgument('bag_path', default_value='/datasets/tum_desk'),
        DeclareLaunchArgument('rviz_config', default_value='/ros2_ws/src/llm_ros/config/llm.rviz'),

        # Topic routing
        DeclareLaunchArgument('rgb_image', default_value=''),
        DeclareLaunchArgument('rgb_info', default_value=''),
        DeclareLaunchArgument('left_image', default_value=''),
        DeclareLaunchArgument('right_image', default_value=''),
        DeclareLaunchArgument('left_info', default_value=''),
        DeclareLaunchArgument('right_info', default_value=''),
        DeclareLaunchArgument('baseline', default_value='0.0'),

        # Bag + topic setup
        OpaqueFunction(function=setup_tum_topics_and_bag),

        # Static TF: map → camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_camera_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera'],
            output='screen'
        ),

        # Core pipeline node
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
            }]
        ),

        # RViz viewer
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
