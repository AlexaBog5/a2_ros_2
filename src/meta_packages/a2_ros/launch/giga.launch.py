"""
Giga autonomy launch: Grid exploration with YOLO target interception and map saving.

Starts:
  - terrainAnalysis + terrainAnalysisExt (mapping)
  - localPlanner + pathFollower (locomotion obstacle-avoidance)
  - far_planner (global visibility-graph planner)
  - grid_goal_manager (dynamic grid exploration, investigate point interception, return home)
  - YOLO object_detection + detection_processor (optional, enabled by default)
  - RViz (optional)

Usage:
  a2 sim --scene scene_test_meshes.xml 


  ros2 launch a2_ros giga.launch.py use_sim_time:=true sim_detection:=true exploration_timeout:=60.0 x_min:=1.0 x_max:=10.0 y_min:=-8.0 y_max:=8.0 spacing:=1.5 

"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    a2_ros_dir = get_package_share_directory('a2_ros')
    rviz_path  = os.path.join(a2_ros_dir, 'rviz', 'navigation.rviz')
    far_config = os.path.join(a2_ros_dir, 'config', 'autonomy', 'far_a2.yaml')

    # ---- Launch Arguments ----
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 with navigation config'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    enable_detection_arg = DeclareLaunchArgument(
        'enable_detection',
        default_value='true',
        description='Launch YOLO object_detection + detection_processor nodes',
    )
    sim_detection_arg = DeclareLaunchArgument(
        'sim_detection',
        default_value='false',
        description='Use sim object_detection launch (uncompressed /camera/image_raw)',
    )
    object_detection_classes_arg = DeclareLaunchArgument(
        'object_detection_classes',
        default_value='[11, 24, 25, 74]',
        description='COCO class IDs for YOLO detection',
    )
    detection_csv_arg = DeclareLaunchArgument(
        'detection_csv',
        default_value='/a2_ros/runs/a2_mission/detections.csv',
        description='CSV output path for detection_processor',
    )
    x_min_arg = DeclareLaunchArgument(
        'x_min',
        default_value='1.0',
        description='Minimum X bound (local forward or global map X)'
    )
    x_max_arg = DeclareLaunchArgument(
        'x_max',
        default_value='9.0',
        description='Maximum X bound (local forward or global map X)'
    )
    y_min_arg = DeclareLaunchArgument(
        'y_min',
        default_value='-4.0',
        description='Minimum Y bound (local right or global map Y)'
    )
    y_max_arg = DeclareLaunchArgument(
        'y_max',
        default_value='4.0',
        description='Maximum Y bound (local left or global map Y)'
    )
    use_local_frame_arg = DeclareLaunchArgument(
        'use_local_frame',
        default_value='true',
        description='Generate grid relative to robot current pose if true'
    )
    exploration_timeout_arg = DeclareLaunchArgument(
        'exploration_timeout',
        default_value='60.0',
        description='Global exploration timeout in seconds'
    )
    spacing_arg = DeclareLaunchArgument(
        'spacing',
        default_value='2.0',
        description='Spacing between grid points (meters)'
    )

    # ---- Conditional Object Detection Groups ----
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
                "'", LaunchConfiguration('enable_detection'), "' == 'true' and '",
                LaunchConfiguration('sim_detection'), "' == 'true'"
            ])
        ),
        launch_arguments={
            'object_detection_classes': '[11, 24, 25, 74]',
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
                "'", LaunchConfiguration('enable_detection'), "' == 'true' and '",
                LaunchConfiguration('sim_detection'), "' != 'true'"
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

    # ---- Core Navigation & Grid Coverage Nodes ----
    nodes = [
        rviz_arg,
        use_sim_time_arg,
        enable_detection_arg,
        sim_detection_arg,
        object_detection_classes_arg,
        detection_csv_arg,
        x_min_arg,
        x_max_arg,
        y_min_arg,
        y_max_arg,
        use_local_frame_arg,
        exploration_timeout_arg,
        spacing_arg,

        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),

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

        # ---- terrain analysis ext (global map for far_planner) ----
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

        # ---- local planner (obstacle avoidance + path following) ----
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
                'maxSpeed':            0.8,
                'dirWeight':           0.1,
                'dirThre':             90.0,
                'dirToVehicle':        False,
                'pathScale':           0.25,
                'minPathScale':        0.5,
                'pathScaleStep':       0.25,
                'pathScaleBySpeed':    False,
                'minPathRange':        1.0,
                'pathRangeStep':       0.5,
                'pathRangeBySpeed':    True,
                'pathCropByGoal':      True,
                'autonomyMode':        True,
                'autonomySpeed':       1.0,
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
                'maxSpeed':         0.8,
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
                'autonomySpeed':    1.0,
                'joyToSpeedDelay':  2.0,
            }],
        ),

        # ---- far_planner (global visibility-graph planner) ----
        Node(
            package='far_planner',
            executable='far_planner',
            name='far_planner',
            output='screen',
            additional_env={'QT_QPA_PLATFORM': 'offscreen'},
            parameters=[far_config],
            remappings=[
                ('/odom_world',         '/state_estimation'),
                ('/terrain_cloud',      '/terrain_map_ext'),
                ('/scan_cloud',         '/registered_scan'),
                ('/terrain_local_cloud','/terrain_map'),
                ('/goal_point',         '/far_planner/goal_point'),
            ],
        ),

        # ---- grid_goal_manager (handles grid coverage, investigate target intercept, and dlio save map) ----
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(a2_ros_dir, 'launch', 'grid_goal_manager.py'),
                '--ros-args',
                '-p', ['x_min:=', LaunchConfiguration('x_min')],
                '-p', ['x_max:=', LaunchConfiguration('x_max')],
                '-p', ['y_min:=', LaunchConfiguration('y_min')],
                '-p', ['y_max:=', LaunchConfiguration('y_max')],
                '-p', ['spacing:=', LaunchConfiguration('spacing')],
                '-p', 'inflation_radius:=0.8',
                '-p', 'goal_timeout:=25.0',
                '-p', 'reach_threshold:=0.8',
                '-p', ['use_local_frame:=', LaunchConfiguration('use_local_frame')],
                '-p', 'save_dir:=/a2_ros/runs/a2_mission',
                '-p', 'map_leaf_size:=0.15',
                '-p', 'auto_save_map:=true',
                '-p', ['exploration_timeout:=', LaunchConfiguration('exploration_timeout')],
                '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
            ],
            name='grid_goal_manager',
            output='screen',
        ),

        # ---- object detection and processor ----
        object_detection_sim,
        object_detection_real,
        detection_processor,

        # ---- RViz with navigation config ----
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_path],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ]

    return LaunchDescription(nodes)
