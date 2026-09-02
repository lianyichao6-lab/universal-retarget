#!/usr/bin/env python3
"""Send one exported AR5/L25 action through luban_framework ROS2.

The default mode is a dry-run. ``--execute`` is required before creating ROS
publishers, and the command publishes one AR5 JointTrajectory point plus one
16-joint L25 Float64MultiArray command.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anydexretarget.luban_action import action_contract


def _read_action(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"arm_positions", "hand_positions", "time_from_start_s"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"action file missing: {', '.join(sorted(missing))}")
        arm = np.asarray(data["arm_positions"], dtype=np.float64)
        hand = np.asarray(data["hand_positions"], dtype=np.float64)
        duration = float(np.asarray(data["time_from_start_s"]).item())
    if arm.shape != (7,) or not np.isfinite(arm).all():
        raise ValueError("arm_positions must be finite with shape (7,)")
    if hand.shape != (16,) or not np.isfinite(hand).all():
        raise ValueError("hand_positions must be finite with shape (16,)")
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("time_from_start_s must be finite and non-negative")
    return {"arm": arm, "hand": hand, "duration": np.asarray(duration)}


def _dry_run(action: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    contract = action_contract()
    report = {
        "planning_only": True,
        "arm_topic": args.arm_topic,
        "hand_topic": args.hand_topic,
        "arm_joint_names": list(contract["arm_joint_names"]),
        "hand_joint_names": list(contract["hand_joint_names"]),
        "arm_positions": action["arm"].tolist(),
        "hand_positions": action["hand"].tolist(),
        "time_from_start_s": float(action["duration"].item()),
    }
    print(json.dumps(report, indent=2))


def _execute(action: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.duration import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python messages are unavailable; source the Luban ROS2 workspace "
            "and run this script with the system ROS Python interpreter"
        ) from exc

    rclpy.init()
    node = rclpy.create_node("anydexretarget_luban_executor")
    try:
        arm_pub = node.create_publisher(JointTrajectory, args.arm_topic, 10)
        hand_pub = node.create_publisher(Float64MultiArray, args.hand_topic, 10)

        arm_msg = JointTrajectory()
        arm_msg.joint_names = list(action_contract()["arm_joint_names"])
        point = JointTrajectoryPoint()
        point.positions = action["arm"].tolist()
        point.time_from_start = Duration(seconds=float(action["duration"].item())).to_msg()
        arm_msg.points = [point]

        hand_msg = Float64MultiArray()
        hand_msg.data = action["hand"].tolist()
        arm_pub.publish(arm_msg)
        hand_pub.publish(hand_msg)
        node.get_logger().info("Published one AR5 trajectory point and one L25 command")
        deadline = time.monotonic() + max(0.2, args.wait_seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--arm-topic", default="/right_arm_controller/joint_trajectory")
    parser.add_argument("--hand-topic", default="/right_hand_controller/commands")
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="actually publish ROS2 messages")
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be non-negative")
    action = _read_action(args.action)
    if not args.execute:
        _dry_run(action, args)
        return 0
    _execute(action, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
