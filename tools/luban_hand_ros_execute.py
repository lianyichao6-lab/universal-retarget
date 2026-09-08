#!/usr/bin/env python3
"""Publish AnyDex hand trajectory frames to a Luban hand controller.

The command is intentionally hand-only: it cannot move an arm. It converts
AnyDex model qpos into Luban's controller order, reports commands by default,
and requires an explicit model-specific confirmation before publishing.
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


def _load_frames(
    path: Path, frame_index: int, all_frames: bool
) -> tuple[list[np.ndarray], list[str]]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must contain a non-empty record list")
    selected = records if all_frames else [records[frame_index]]
    targets: list[np.ndarray] = []
    source_names: list[str] = []
    for record in selected:
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError("trajectory record lacks target")
        qpos = np.asarray(record["target"], dtype=np.float64).reshape(-1)
        if not np.isfinite(qpos).all():
            raise ValueError("trajectory target must contain only finite values")
        targets.append(qpos)
        if not source_names:
            source_names = [str(name) for name in record.get("joint_names", [])]
    return targets, source_names


def _report(
    args: argparse.Namespace, commands: list[np.ndarray], source_names: list[str]
) -> None:
    print(json.dumps({
        "planning_only": not args.execute,
        "hand_model": args.hand_model,
        "frame": "all" if args.all_frames else args.frame,
        "frame_count": len(commands),
        "fps": args.fps if args.all_frames else None,
        "hand_topic": args.hand_topic,
        "controller_joint_names": list(
            hand_active_joint_names(hand_model=args.hand_model, side=args.side)
        ),
        "source_joint_names": source_names,
        "first_command_positions_rad": commands[0].tolist(),
        "last_command_positions_rad": commands[-1].tolist(),
    }, indent=2))


def _publish(args: argparse.Namespace, commands: list[np.ndarray]) -> None:
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
        period = 1.0 / args.fps if args.all_frames else 0.0
        for index, command in enumerate(commands):
            started = time.monotonic()
            message = Float64MultiArray()
            message.data = command.tolist()
            repeats = 1 if args.all_frames else args.publish_count
            for _ in range(repeats):
                publisher.publish(message)
                rclpy.spin_once(node, timeout_sec=0.01)
                if not args.all_frames:
                    time.sleep(args.publish_period_s)
            if args.all_frames and index + 1 < len(commands):
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        node.get_logger().info(
            f"Published {args.hand_model.upper()} {len(commands)} frame(s) "
            f"to {args.hand_topic}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--fps", type=float, default=15.0)
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
    if (
        args.discovery_timeout < 0
        or args.publish_count < 1
        or args.publish_period_s < 0
        or args.fps <= 0
    ):
        parser.error("timeouts and periods must be non-negative; counts and fps must be positive")
    expected_confirmation = f"{args.hand_model.upper()}_HAND_CLEAR"
    if args.execute and args.confirm != expected_confirmation:
        parser.error(f"--execute requires --confirm {expected_confirmation}")
    qposes, source_names = _load_frames(args.trajectory, args.frame, args.all_frames)
    commands = [
        hand_qpos_to_luban_active(qpos, hand_model=args.hand_model)
        for qpos in qposes
    ]
    _report(args, commands, source_names)
    if args.execute:
        _publish(args, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
