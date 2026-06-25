"""
Mega autonomy launch: TARE + FAR planners with waypoint mux.

Starts terrain analysis, local planner, path follower, TARE exploration,
FAR navigation, waypoint_mux, and optionally object detection + detection_processor.

Prerequisites (sim.launch.py + a2_bridge):
  /state_estimation, /registered_scan, /clock

Usage:
  ros2 launch a2_ros mega.launch.py rviz:=true use_sim_time:=true
  ros2 launch a2_ros mega.launch.py use_sim_time:=true enable_detection:=true sim_detection:=true

Switch planner at runtime:
  ros2 topic pub --once /planner/select std_msgs/msg/String "{data: 'far'}"
  ros2 topic pub --once /planner/select std_msgs/msg/String "{data: 'tare'}"
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description_dir = get_package_share_directory('a2_description')
    a2_ros_dir      = get_package_share_directory('a2_ros')
    rviz_path        = os.path.join(a2_ros_dir, 'rviz', 'exploration.rviz')
    tare_config      = os.path.join(a2_ros_dir, 'config', 'autonomy', 'tare_a2.yaml')
    far_config       = os.path.join(a2_ros_dir, 'config', 'autonomy', 'far_a2.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    enable_detection_arg = DeclareLaunchArgument(
        'enable_detection',
        default_value='false',
        description='Launch YOLO object_detection + detection_processor nodes',
    )
    sim_detection_arg = DeclareLaunchArgument(
        'sim_detection',
        default_value='false',
        description='Use sim object_detection launch (uncompressed /camera/image_raw)',
    )
    object_detection_classes_arg = DeclareLaunchArgument(
        'object_detection_classes',
        default_value='[39, 25]',
        description='COCO class IDs for YOLO detection',
    )
    detection_csv_arg = DeclareLaunchArgument(
        'detection_csv',
        default_value='/tmp/a2_mission/detections.csv',
        description='CSV output path for detection_processor',
    )

    object_detection_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('object_detection'),
                'launch',
                'object_detection.launch.py',
            ])
        ),
        condition=IfCondition(
            PythonExpression([
                "'",
                LaunchConfiguration('enable_detection'),
                "' == 'true' and '",
                LaunchConfiguration('sim_detection'),
                "' == 'true'",
            ])
        ),
        launch_arguments={
            'object_detection_classes': LaunchConfiguration('object_detection_classes'),
            'lidar_topic': '/front_lidar/points',
            'input_camera_name': '/camera',
        }.items(),
    )

    object_detection_real = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('object_detection'),
                'launch',
                'object_detection_real.launch.py',
            ])
        ),
        condition=IfCondition(
            PythonExpression([
                "'",
                LaunchConfiguration('enable_detection'),
                "' == 'true' and '",
                LaunchConfiguration('sim_detection'),
                "' != 'true'",
            ])
        ),
        launch_arguments={
            'object_detection_classes': LaunchConfiguration('object_detection_classes'),
            'lidar_topic': '/front_lidar/points',
            'input_camera_name': '/camera',
            'debayer_image': 'false',
        }.items(),
    )

    detection_processor = Node(
        package='a2_orchestrator',
        executable='detection_processor',
        name='detection_processor',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_detection')),
        parameters=[{
            'detection_info_topic': '/detection_info',
            'investigate_point_topic': '/investigate_point',
            'detection_enable_topic': '/detection/enable',
            'detection_save_topic': '/detection/save',
            'map_frame': 'map',
            'output_csv': LaunchConfiguration('detection_csv'),
        }],
    )

    nodes = [
        rviz_arg,
        use_sim_time_arg,
        enable_detection_arg,
        sim_detection_arg,
        object_detection_classes_arg,
        detection_csv_arg,
        SetParameter(
            name='use_sim_time',
            value=ParameterValue(use_sim_time, value_type=bool),
        ),

        # ---- terrain analysis (local map) ----
        Node(
            package='terrain_analysis',
            executable='terrainAnalysis',
            name='terrainAnalysis',
            output='screen',
            parameters=[{
                'scanVoxelSize':       0.05,
                'decayTime':           10.0,
                'noDecayDis':          0.0,
                'clearingDis':         8.0,
                'useSorting':          True,
                'quantileZ':           0.25,
                'considerDrop':        True,
                'limitGroundLift':     True,
                'maxGroundLift':       0.25,
                'clearDyObs':          False,
                'minDyObsDis':         0.3,
                'minDyObsAngle':       0.0,
                'minDyObsRelZ':        -0.5,
                'absDyObsRelZThre':    0.2,
                'minDyObsVFOV':        -16.0,
                'maxDyObsVFOV':        16.0,
                'minDyObsPointNum':    1,
                'noDataObstacle':      False,
                'noDataBlockSkipNum':  0,
                'minBlockPointNum':    10,
                'vehicleHeight':       0.5,
                'voxelPointUpdateThre': 100,
                'voxelTimeUpdateThre': 2.0,
                'minRelZ':             -1.0,
                'maxRelZ':             1.0,
                'disRatioZ':           0.2,
            }],
        ),

        # ---- terrain analysis ext (global map) ----
        Node(
            package='terrain_analysis_ext',
            executable='terrainAnalysisExt',
            name='terrainAnalysisExt',
            output='screen',
            parameters=[{
                'scanVoxelSize':        0.1,
                'decayTime':            10.0,
                'noDecayDis':           0.0,
                'clearingDis':          30.0,
                'useSorting':           True,
                'quantileZ':            0.25,
                'vehicleHeight':        0.5,
                'voxelPointUpdateThre': 100,
                'voxelTimeUpdateThre':  2.0,
                'lowerBoundZ':          -1.0,
                'upperBoundZ':          1.0,
                'disRatioZ':            0.1,
                'checkTerrainConn':     True,
                'terrainUnderVehicle':  -0.75,
                'terrainConnThre':      0.5,
                'ceilingFilteringThre': 2.0,
                'localTerrainMapRadius': 4.0,
            }],
        ),
        # ---- local planner ----
        Node(
            package='local_planner',
            executable='localPlanner',
            name='localPlanner',
            output='screen',
            parameters=[{
                'pathFolder':          get_package_share_directory('local_planner') + '/paths',
                'vehicleLength':       0.65,
                'vehicleWidth':        0.40,
                'sensorOffsetX':       0.0,
                'sensorOffsetY':       0.0,
                'twoWayDrive':         False,
                'laserVoxelSize':      0.05,
                'terrainVoxelSize':    0.2,
                'useTerrainAnalysis':  True,
                'checkObstacle':       True,
                'checkRotObstacle':    True,
                'adjacentRange':       3.5,
                'obstacleHeightThre':  0.25,
                'groundHeightThre':    0.1,
                'costHeightThre':      0.1,
                'costScore':           0.02,
                'useCost':             False,
                'pointPerPathThre':    2,
                'minRelZ':             -0.5,
                'maxRelZ':             0.8,
                'maxSpeed':            0.5,
                'dirWeight':           0.1,
                'dirThre':             90.0,
                'dirToVehicle':        False,
                'pathScale':           1.0,
                'minPathScale':        0.75,
                'pathScaleStep':       0.25,
                'pathScaleBySpeed':    True,
                'minPathRange':        1.0,
                'pathRangeStep':       0.5,
                'pathRangeBySpeed':    True,
                'pathCropByGoal':      True,
                'autonomyMode':        True,
                'autonomySpeed':       2.0,
                'joyToSpeedDelay':     2.0,
                'joyToCheckObstacleDelay': 5.0,
                'goalClearRange':      0.4,
                'goalX':               0.0,
                'goalY':               0.0,
            }],
        ),

        Node(
            package='local_planner',
            executable='pathFollower',
            name='pathFollower',
            output='screen',
            parameters=[{
                'sensorOffsetX':    0.0,
                'sensorOffsetY':    0.0,
                'pubSkipNum':       1,
                'twoWayDrive':      False,
                'lookAheadDis':     0.4,
                'yawRateGain':      10.0,
                'stopYawRateGain':  8.0,
                'maxYawRate':       45.0,
                'maxSpeed':         0.5,
                'maxAccel':         2.0,
                'switchTimeThre':   1.0,
                'dirDiffThre':      0.1,
                'stopDisThre':      0.3,
                'slowDwnDisThre':   0.6,
                'useInclRateToSlow': False,
                'inclRateThre':     120.0,
                'slowRate1':        0.25,
                'slowRate2':        0.5,
                'slowTime1':        2.0,
                'slowTime2':        2.0,
                'useInclToStop':    False,
                'inclThre':         45.0,
                'stopTime':         5.0,
                'noRotAtStop':      False,
                'noRotAtGoal':      True,
                'autonomyMode':     True,
                'autonomySpeed':    2.0,
                'joyToSpeedDelay':  2.0,
            }],
        ),

        # ---- TARE planner (autonomous exploration) ----
        Node(
            package='tare_planner',
            executable='tare_planner_node',
            name='tare_planner_node',
            output='screen',
            parameters=[
                tare_config,
                {
                    'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                },
            ],
        ),

        # ---- far_planner (global visibility-graph planner) ----
        Node(
            package='far_planner',
            executable='far_planner',
            name='far_planner',
            output='screen',
            # Run headless: no X display in container/SSH, so force Qt offscreen
            # to avoid the xcb plugin aborting (SIGABRT). Planning still works;
            # use RViz instead of the FAR Planner GUI for visualization.
            additional_env={'QT_QPA_PLATFORM': 'offscreen'},
            parameters=[far_config],
            remappings=[
                ('/odom_world',         '/state_estimation'),
                ('/terrain_cloud',      '/terrain_map_ext'),
                ('/scan_cloud',         '/registered_scan'),
                ('/terrain_local_cloud','/terrain_map'),
                ('/way_point',          '/far/way_point'),
            ],
        ),

        # ---- waypoint mux (TARE or FAR -> /way_point) ----
        Node(
            package='a2_orchestrator',
            executable='waypoint_mux',
            name='waypoint_mux',
            output='screen',
            parameters=[{
                'default_source': 'tare',
                'tare_waypoint_topic': '/tare/way_point',
                'far_waypoint_topic': '/far/way_point',
                'output_waypoint_topic': '/way_point',
                'select_topic': '/planner/select',
            }],
        ),


        # ---- object detection (optional; gated at runtime via /detection/enable) ----
        object_detection_sim,
        object_detection_real,
        detection_processor,

        # ---- RViz ----
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_path],
            parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ]

    return LaunchDescription(nodes)
