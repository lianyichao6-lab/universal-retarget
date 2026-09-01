#!/usr/bin/env python3
"""Publish an offline L25 retarget trajectory as ROS 2 JointState messages."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# This is the qpos order written by the direct-qpos L25 MuJoCo renderer.
L25_QPOS_JOINTS = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
]
MIMIC_JOINTS = {"thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"}
INDEPENDENT_INDICES = [i for i, name in enumerate(L25_QPOS_JOINTS) if name not in MIMIC_JOINTS]
INDEPENDENT_JOINTS = [L25_QPOS_JOINTS[i] for i in INDEPENDENT_INDICES]


def load_trajectory(path: Path) -> list[list[float]]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Trajectory must be a non-empty teleop_sim.py pickle list")
    trajectory = []
    for frame, record in enumerate(records):
        qpos = record.get("target") if isinstance(record, dict) else None
        if qpos is None or len(qpos) != len(L25_QPOS_JOINTS):
            raise ValueError(f"Frame {frame} does not contain a 21-value L25 target")
        trajectory.append([float(qpos[index]) for index in INDEPENDENT_INDICES])
    return trajectory


class TrajectoryPlayer(Node):
    def __init__(self, trajectory: list[list[float]], fps: float, loop: bool) -> None:
        super().__init__("l25_offline_trajectory_player")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.trajectory = trajectory
        self.loop = loop
        self.frame = 0
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(
            f"Playing {len(trajectory)} L25 frames at {fps:.1f} Hz; "
            f"publishing {len(INDEPENDENT_JOINTS)} independent joints."
        )

    def publish_frame(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = INDEPENDENT_JOINTS
        message.position = self.trajectory[self.frame]
        self.publisher.publish(message)
        self.frame += 1
        if self.frame == len(self.trajectory):
            if self.loop:
                self.frame = 0
            else:
                self.timer.cancel()
                self.get_logger().info("Trajectory complete; retaining the final pose.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline L25 trajectory player for RViz")
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    trajectory = load_trajectory(args.trajectory)
    rclpy.init()
    node = TrajectoryPlayer(trajectory, args.fps, not args.no_loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
