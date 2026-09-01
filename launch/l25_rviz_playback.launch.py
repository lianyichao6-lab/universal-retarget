"""Offline RViz playback for an L25 AnyDexRetarget trajectory.

This launch file starts no LinkerHand SDK, ros2_control hardware, or actuator
node. It only runs robot_state_publisher, a JointState file player, and RViz.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "linkerhand_l25" / "right" / "linkerhand_l25_right.urdf"
PLAYER_PATH = ROOT / "tools" / "l25_rviz_playback.py"
MARKER_PATH = ROOT / "tools" / "hug_skeleton_marker.py"
RVIZ_CONFIG_PATH = ROOT / "launch" / "l25_playback.rviz"
DEFAULT_TRAJECTORY = ROOT / "outputs" / "l25" / "l25_vector_high_sim.pkl"


def _robot_description() -> str:
    description = URDF_PATH.read_text()
    mesh_dir = (URDF_PATH.parent / "meshes").as_uri()
    return description.replace('filename="meshes/', f'filename="{mesh_dir}/')


def generate_launch_description() -> LaunchDescription:
    trajectory = LaunchConfiguration("trajectory")
    fps = LaunchConfiguration("fps")
    return LaunchDescription([
        DeclareLaunchArgument("trajectory", default_value=str(DEFAULT_TRAJECTORY)),
        DeclareLaunchArgument("fps", default_value="30.0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": _robot_description()}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "world", "hand_base_link"],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(RVIZ_CONFIG_PATH)],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
        ExecuteProcess(
            cmd=["/usr/bin/python3", str(PLAYER_PATH), "--trajectory", trajectory, "--fps", fps],
            output="screen",
        ),
        ExecuteProcess(
            cmd=["/usr/bin/python3", str(MARKER_PATH), "--trajectory", trajectory, "--fps", fps],
            output="screen",
        ),
    ])
