"""Launch mission_orchestrator with configurable pre-explore FAR waypoint."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_dir = get_package_share_directory('a2_orchestrator')
    defaults = os.path.join(pkg_dir, 'config', 'mission_defaults.yaml')

    use_pre_explore_arg = DeclareLaunchArgument(
        'use_pre_explore_goal',
        default_value='true',
        description='Navigate to pre_explore_goal with FAR before TARE exploration',
    )
    pre_x_arg = DeclareLaunchArgument(
        'pre_explore_goal_x',
        default_value='2.0',
        description='Pre-explore goal X in map frame (m)',
    )
    pre_y_arg = DeclareLaunchArgument(
        'pre_explore_goal_y',
        default_value='0.0',
        description='Pre-explore goal Y in map frame (m)',
    )

    return LaunchDescription([
        use_pre_explore_arg,
        pre_x_arg,
        pre_y_arg,
        Node(
            package='a2_orchestrator',
            executable='mission_orchestrator',
            name='mission_orchestrator',
            output='screen',
            parameters=[
                defaults,
                {
                    'use_pre_explore_goal': ParameterValue(
                        LaunchConfiguration('use_pre_explore_goal'), value_type=bool
                    ),
                    'pre_explore_goal_x': ParameterValue(
                        LaunchConfiguration('pre_explore_goal_x'), value_type=float
                    ),
                    'pre_explore_goal_y': ParameterValue(
                        LaunchConfiguration('pre_explore_goal_y'), value_type=float
                    ),
                },
            ],
        ),
    ])
