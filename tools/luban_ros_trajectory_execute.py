#!/usr/bin/env python3
"""Publish a synchronized AR5/L25 trajectory to Luban ROS2.

The input is an offline ``.npz`` with ``arm_positions`` (N, 7),
``hand_positions`` (N, 16), and ``timestamps`` (N,) in seconds from start.
Without ``--execute`` this only validates and prints a planning report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anydexretarget.luban_action import action_contract


def read_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"arm_positions", "hand_positions", "timestamps"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"trajectory file missing: {', '.join(sorted(missing))}")
        arm = np.asarray(data["arm_positions"], dtype=np.float64)
        hand = np.asarray(data["hand_positions"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 7 or not np.isfinite(arm).all():
        raise ValueError(f"arm_positions must be finite shape (N,7), got {arm.shape}")
    if hand.shape != (len(arm), 16) or not np.isfinite(hand).all():
        raise ValueError(f"hand_positions must be finite shape ({len(arm)},16), got {hand.shape}")
    if timestamps.shape != (len(arm),) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must have shape (N,) and contain finite values")
    if len(timestamps) == 0 or timestamps[0] < 0 or np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be non-empty, non-negative, and non-decreasing")
    return {"arm": arm, "hand": hand, "timestamps": timestamps}


def _dry_run(trajectory: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    contract = action_contract()
    report = {
        "planning_only": True,
        "frames": int(len(trajectory["timestamps"])),
        "duration_s": float(trajectory["timestamps"][-1]),
        "arm_topic": args.arm_topic,
        "hand_topic": args.hand_topic,
        "arm_joint_names": list(contract["arm_joint_names"]),
        "hand_joint_names": list(contract["hand_joint_names"]),
    }
    print(json.dumps(report, indent=2))


def _execute(trajectory: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.duration import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError("source the Luban ROS2 workspace and use its Python interpreter") from exc

    rclpy.init()
    node = rclpy.create_node("anydexretarget_luban_trajectory_executor")
    arm_pub = node.create_publisher(JointTrajectory, args.arm_topic, 10)
    hand_pub = node.create_publisher(Float64MultiArray, args.hand_topic, 10)
    arm_msg = JointTrajectory(joint_names=list(action_contract()["arm_joint_names"]))
    for arm, stamp in zip(trajectory["arm"], trajectory["timestamps"]):
        point = JointTrajectoryPoint(positions=arm.tolist())
        point.time_from_start = Duration(seconds=float(stamp)).to_msg()
        arm_msg.points.append(point)
    arm_pub.publish(arm_msg)
    start = time.monotonic()
    frame = 0
    try:
        while rclpy.ok() and frame < len(trajectory["timestamps"]):
            elapsed = time.monotonic() - start
            while frame < len(trajectory["timestamps"]) and trajectory["timestamps"][frame] <= elapsed:
                hand_pub.publish(Float64MultiArray(data=trajectory["hand"][frame].tolist()))
                frame += 1
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm-topic", default="/right_arm_controller/joint_trajectory")
    parser.add_argument("--hand-topic", default="/right_hand_controller/commands")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    trajectory = read_trajectory(args.trajectory)
    if args.execute:
        _execute(trajectory, args)
    else:
        _dry_run(trajectory, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
