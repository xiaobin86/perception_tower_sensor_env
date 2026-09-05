"""One-click launch for sensor_env: LiDAR + camera + turntable.

Usage:
    ros2 launch perception_tower_sensor sensor_env.launch.py
    ros2 launch perception_tower_sensor sensor_env.launch.py turntable_port:=/dev/ttyUSB1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    turntable_port = LaunchConfiguration("turntable_port")
    params_file = LaunchConfiguration("params_file")

    default_params = PathJoinSubstitution(
        [FindPackageShare("perception_tower_sensor"), "config", "turntable_params.yaml"]
    )

    # LiDAR launch
    rslidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rslidar_sdk"), "launch", "start.py"])
        )
    )

    # Camera launch
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("orbbec_camera"), "launch", "gemini_330_series.launch.py"])
        )
    )

    # Turntable node
    turntable_node = Node(
        package="perception_tower_sensor",
        executable="turntable_node",
        name="turntable_node",
        output="screen",
        parameters=[params_file, {"serial_port": turntable_port}],
    )

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
        rslidar_launch,
        orbbec_launch,
        turntable_node,
    ])
