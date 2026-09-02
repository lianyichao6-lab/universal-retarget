#!/usr/bin/env python3
"""Record one Luban AR5 JointTrajectory message as a portable NPZ."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/right_arm_controller/joint_trajectory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    import rclpy
    from trajectory_msgs.msg import JointTrajectory

    received: list[JointTrajectory] = []
    rclpy.init()
    node = rclpy.create_node("anydexretarget_luban_trajectory_recorder")
    node.create_subscription(JointTrajectory, args.topic, received.append, 10)
    deadline = time.monotonic() + args.timeout_s
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not received:
        raise TimeoutError(f"no JointTrajectory received on {args.topic} within {args.timeout_s:.1f}s")

    message = received[0]
    names = np.asarray([str(name) for name in message.joint_names])
    if names.size != 7 or set(names.tolist()) != {f"r_joint_{i}" for i in range(1, 8)}:
        raise ValueError(f"expected Luban AR5 joint names, got {names.tolist()}")
    positions = np.asarray([list(point.positions) for point in message.points], dtype=np.float64)
    timestamps = np.asarray(
        [float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9 for point in message.points],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 7 or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError(f"trajectory positions must be finite with shape (N,7), got {positions.shape}")
    if timestamps.shape != (len(positions),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("trajectory timestamps must be finite, non-decreasing and aligned")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joint_names=names, positions=positions, timestamps=timestamps)
    print(f"Recorded {len(positions)} AR5 trajectory points from {args.topic}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
