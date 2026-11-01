"""Launch file for starting the muzammil_controller node with YAML parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("muzammil_controller")
    default_params = os.path.join(pkg_share, "config", "controller_params.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to ROS 2 parameter YAML file for muzammil_controller.",
    )

    controller_node = Node(
        package="muzammil_controller",
        executable="controller",
        name="muzammil_f1tenth_controller",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    return LaunchDescription([params_arg, controller_node])
