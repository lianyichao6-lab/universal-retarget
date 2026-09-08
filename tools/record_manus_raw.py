#!/usr/bin/env python3
"""Record Luban MANUS frames from ROS 2 into a portable NumPy archive."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from manus_msgs.msg import ManusState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class ManusRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("universal_retarget_manus_recorder")
        self.args = args
        self.frames: list[tuple[float, np.ndarray, np.ndarray, int, np.ndarray]] = []
        self.finished = False
        self.last_sample_s = float("-inf")
        self.started_s = time.monotonic()
        self.invalid_frames = 0
        self.create_subscription(
            ManusState, args.topic, self._on_state, qos_profile_sensor_data
        )

    def _on_state(self, message: ManusState) -> None:
        hand = getattr(message, self.args.side)
        if not hand.valid or not hand.wrist_valid:
            self.invalid_frames += 1
            return
        timestamp_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )
        if timestamp_s - self.last_sample_s < 1.0 / self.args.rate_hz:
            return
        wrist = np.asarray(
            (hand.wrist.position.x, hand.wrist.position.y, hand.wrist.position.z),
            dtype=np.float32,
        )
        keypoints = np.asarray(
            [(pose.position.x, pose.position.y, pose.position.z) for pose in hand.keypoints],
            dtype=np.float32,
        )
        ergonomics = np.asarray(hand.ergonomics, dtype=np.float32)
        if keypoints.shape != (25, 3) or not np.isfinite(wrist).all():
            self.invalid_frames += 1
            return
        if ergonomics.shape != (20,):
            ergonomics = np.full(20, np.nan, dtype=np.float32)
        self.frames.append(
            (timestamp_s, wrist, keypoints, int(hand.keypoint_mask), ergonomics)
        )
        self.last_sample_s = timestamp_s
        if len(self.frames) == 1:
            self.get_logger().info(
                f"Recording {self.args.side} MANUS frames from {self.args.topic}"
            )
        if self.args.max_frames and len(self.frames) >= self.args.max_frames:
            self.finished = True
        if (
            self.args.duration > 0
            and time.monotonic() - self.started_s >= self.args.duration
        ):
            self.finished = True

    def save(self) -> None:
        if not self.frames:
            raise RuntimeError(
                "No valid MANUS frames received; check the topic, side and glove connection"
            )
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.args.output,
            schema_version=np.asarray(1, dtype=np.int64),
            source=np.asarray("manus_ros2"),
            topic=np.asarray(self.args.topic),
            handedness=np.asarray(self.args.side),
            timestamps_s=np.asarray([frame[0] for frame in self.frames], dtype=np.float64),
            wrists=np.stack([frame[1] for frame in self.frames]),
            keypoints_25=np.stack([frame[2] for frame in self.frames]),
            keypoint_masks=np.asarray([frame[3] for frame in self.frames], dtype=np.uint32),
            ergonomics=np.stack([frame[4] for frame in self.frames]),
        )
        self.get_logger().info(
            f"Saved {len(self.frames)} MANUS frames to {self.args.output} "
            f"(invalid frames skipped: {self.invalid_frames})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/manus/state")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 records until Ctrl-C")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate_hz <= 0 or args.duration < 0 or args.max_frames < 0:
        raise ValueError("rate-hz must be positive; duration and max-frames cannot be negative")
    rclpy.init()
    node = ManusRecorder(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
