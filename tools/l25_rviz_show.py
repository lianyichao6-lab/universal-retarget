#!/usr/bin/env python3
"""Show the L25 hand in an already-running RViz, without the HUG environment.

Resolves the URDF mesh paths, launches ``robot_state_publisher``, and publishes
a neutral ``/joint_states`` so the hand appears at the world origin. Run this
after the perception RViz is up, then add a RobotModel display in RViz.

Usage:
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 tools/l25_rviz_show.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "linkerhand_l25" / "right" / "linkerhand_l25_right.urdf"

# Mirrors l25_rviz_playback.L25_QPOS_JOINTS minus the five mimic joints.
INDEPENDENT_JOINTS = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
]


def _resolved_urdf() -> str:
    mesh_dir = (URDF_PATH.parent / "meshes").as_uri()
    return URDF_PATH.read_text().replace('filename="meshes/', f'filename="{mesh_dir}/')


def _write_params(path: Path) -> None:
    body = "".join(f"      {line}\n" for line in _resolved_urdf().splitlines())
    path.write_text(
        "l25_state_publisher:\n"
        "  ros__parameters:\n"
        "    robot_description: |\n"
        f"{body}",
        encoding="utf-8",
    )


def _load_plan_qpos(plan_path: Path) -> list[float]:
    data = np.load(plan_path, allow_pickle=True)
    names = [str(n) for n in data["robot_joint_names"]]
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    value_by_name = dict(zip(names, qpos))
    return [float(value_by_name.get(name, 0.0)) for name in INDEPENDENT_JOINTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        help="Optional L25 collision-aware plan .npz; publish its qpos instead of neutral.")
    args = parser.parse_args()
    if not shutil.which("ros2"):
        sys.exit("ros2 not on PATH; run `source /opt/ros/jazzy/setup.bash` first.")
    qpos = _load_plan_qpos(args.plan) if args.plan else [0.0] * len(INDEPENDENT_JOINTS)

    params_file = Path(tempfile.gettempdir()) / "l25_rsp_params.yaml"
    _write_params(params_file)

    proc = subprocess.Popen(
        [
            "ros2", "run", "robot_state_publisher", "robot_state_publisher",
            "--ros-args", "-r", "__node:=l25_state_publisher",
            "--params-file", str(params_file),
        ],
    )

    rclpy.init()
    node = Node("l25_neutral_joints")
    pub = node.create_publisher(JointState, "/joint_states", 10)

    # Anchor the hand's root link in the RViz fixed frame (world).
    static_broadcaster = StaticTransformBroadcaster(node)
    anchor = TransformStamped()
    anchor.header.stamp = node.get_clock().now().to_msg()
    anchor.header.frame_id = "world"
    anchor.child_frame_id = "hand_base_link"
    anchor.transform.translation.x = 0.0
    anchor.transform.translation.y = 0.0
    anchor.transform.translation.z = 0.0
    anchor.transform.rotation.w = 1.0
    static_broadcaster.sendTransform(anchor)

    def publish_qpos() -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = INDEPENDENT_JOINTS
        msg.position = qpos
        pub.publish(msg)

    publish_qpos()
    timer = node.create_timer(1.0, publish_qpos)
    pose_desc = f"plan qpos from {args.plan}" if args.plan else "neutral pose"
    node.get_logger().info(
        f"L25 hand shown at {pose_desc} (world origin). In RViz: Add -> By display "
        "type -> rviz_default_plugins -> RobotModel. Ctrl-C to stop."
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
