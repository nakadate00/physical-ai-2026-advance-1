"""4 ノードをまとめて起動する launch ファイル."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    seed = LaunchConfiguration("seed")
    return LaunchDescription([
        DeclareLaunchArgument("seed", default_value="20261102"),
        Node(package="agripioneer_ros", executable="simulator_node",
             name="simulator_node", output="screen",
             parameters=[{"seed": seed}]),
        Node(package="agripioneer_ros", executable="perception_node",
             name="perception_node", output="screen"),
        Node(package="agripioneer_ros", executable="planner_node",
             name="planner_node", output="screen"),
        Node(package="agripioneer_ros", executable="controller_node",
             name="controller_node", output="screen"),
    ])
