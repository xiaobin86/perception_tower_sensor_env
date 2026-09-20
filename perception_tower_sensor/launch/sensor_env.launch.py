"""One-click launch for sensor_env: LiDAR + camera + turntable.

Usage:
    ros2 launch perception_tower_sensor sensor_env.launch.py
    ros2 launch perception_tower_sensor sensor_env.launch.py turntable_port:=/dev/ttyUSB1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    turntable_port = LaunchConfiguration("turntable_port")
    params_file = LaunchConfiguration("params_file")
    use_fairy = LaunchConfiguration("use_fairy")
    enable_colored_pc = LaunchConfiguration("enable_colored_pc")
    ordered_pc = LaunchConfiguration("ordered_pc")

    default_params = PathJoinSubstitution(
        [FindPackageShare("perception_tower_sensor"), "config", "turntable_params.yaml"]
    )

    # LiDAR node (bypass upstream launch to avoid forced rviz2)
    rslidar_node = Node(
        package="rslidar_sdk",
        executable="rslidar_sdk_node",
        name="rslidar_sdk_node",
        namespace="rslidar_sdk",
        output="screen",
        parameters=[{"config_path": "/opt/fairy_ws/config/config.yaml"}],
        condition=IfCondition(use_fairy),
        on_exit=Shutdown(),
    )
    rslidar_node_delayed = TimerAction(period=10.0, actions=[rslidar_node])

    # Camera launch (start first so it can claim UVC resources before LiDAR starts)
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("orbbec_camera"), "launch", "gemini_330_series.launch.py"])
        ),
        launch_arguments={
            "enable_colored_point_cloud": enable_colored_pc,
            "ordered_pc": ordered_pc,
        }.items(),
    )

    # Turntable node (delayed until camera and LiDAR are stable)
    turntable_node = Node(
        package="perception_tower_sensor",
        executable="turntable_node",
        name="turntable_node",
        output="screen",
        parameters=[params_file, {"serial_port": turntable_port}],
        on_exit=Shutdown(),
    )
    turntable_node_delayed = TimerAction(period=15.0, actions=[turntable_node])

    return LaunchDescription([
        DeclareLaunchArgument(
            "turntable_port",
            default_value="/dev/ttyUSB0",
            description="Serial port for turntable STM32.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to turntable parameter YAML.",
        ),
        DeclareLaunchArgument(
            "use_fairy",
            default_value="true",
            description="Launch the RoboSense Fairy LiDAR driver.",
        ),
        DeclareLaunchArgument(
            "enable_colored_pc",
            default_value="true",
            description="Publish the RGB cloud on /camera/depth_registered/points (D2C-aligned to the color "
                        "camera; requires identical color and depth resolutions, else frames are dropped).",
        ),
        DeclareLaunchArgument(
            "ordered_pc",
            default_value="true",
            description="Keep the organized WxH grid so point index i maps to color pixel (i%W, i/W); "
                        "with false the cloud is compacted to width=N, height=1.",
        ),
        orbbec_launch,
        rslidar_node_delayed,
        turntable_node_delayed,
    ])
