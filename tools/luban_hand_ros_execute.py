#!/usr/bin/env python3
"""Publish one AnyDex hand trajectory frame to a Luban hand controller.

This tool is intentionally hand-only: it cannot move an arm. It converts
AnyDex model qpos to Luban's controller order, reports the exact command by
default, and requires an explicit model-specific confirmation before publish.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from anydexretarget.luban_contract import (
    hand_active_joint_names,
    hand_qpos_to_luban_active,
    normalize_hand_model,
)


def _load_frame(path: Path, frame_index: int) -> tuple[np.ndarray, list[str]]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must contain a non-empty record list")
    try:
        record = records[frame_index]
    except IndexError as exc:
        raise ValueError(f"frame index {frame_index} is outside trajectory") from exc
    if not isinstance(record, dict) or "target" not in record:
        raise ValueError("trajectory record lacks target")
    qpos = np.asarray(record["target"], dtype=np.float64)
    names = [str(name) for name in record.get("joint_names", [])]
    if not np.isfinite(qpos).all():
        raise ValueError("trajectory target must contain only finite values")
    return qpos, names


def _report(args: argparse.Namespace, command: np.ndarray, source_names: list[str]) -> None:
    print(json.dumps({
        "planning_only": not args.execute,
        "hand_model": args.hand_model,
        "frame": args.frame,
        "hand_topic": args.hand_topic,
        "controller_joint_names": list(hand_active_joint_names(hand_model=args.hand_model, side=args.side)),
        "source_joint_names": source_names,
        "command_positions_rad": command.tolist(),
    }, indent=2))


def _publish(args: argparse.Namespace, command: np.ndarray) -> None:
    try:
        import rclpy
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python messages are unavailable; source Luban and run with its system Python"
        ) from exc
    rclpy.init()
    node = rclpy.create_node("anydexretarget_luban_hand_executor")
    try:
        publisher = node.create_publisher(Float64MultiArray, args.hand_topic, 10)
        deadline = time.monotonic() + args.discovery_timeout
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() == 0:
            raise TimeoutError(f"No subscriber discovered on {args.hand_topic}")
        message = Float64MultiArray()
        message.data = command.tolist()
        for _ in range(args.publish_count):
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(args.publish_period_s)
        node.get_logger().info(
            f"Published {args.hand_model.upper()} {len(command)}-axis hand command"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--hand-model", choices=("l25", "o30"), required=True)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--hand-topic")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--publish-period-s", type=float, default=0.05)
    args = parser.parse_args()
    args.hand_model = normalize_hand_model(args.hand_model)
    if args.hand_topic is None:
        args.hand_topic = f"/{args.side}_hand_controller/commands"
    if args.discovery_timeout < 0 or args.publish_count < 1 or args.publish_period_s < 0:
        parser.error("timeouts must be non-negative and --publish-count must be positive")
    expected_confirmation = f"{args.hand_model.upper()}_HAND_CLEAR"
    if args.execute and args.confirm != expected_confirmation:
        parser.error(f"--execute requires --confirm {expected_confirmation}")
    qpos, source_names = _load_frame(args.trajectory, args.frame)
    command = hand_qpos_to_luban_active(qpos, hand_model=args.hand_model)
    _report(args, command, source_names)
    if args.execute:
        _publish(args, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
