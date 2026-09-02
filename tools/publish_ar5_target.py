#!/usr/bin/env python3
"""Publish one calibrated AR5 flange target to Luban's pose planner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from anydexretarget.luban_arm import arm_flange_pose_xyzw


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        if "T_robot_base_arm_flange" in data:
            target = np.asarray(data["T_robot_base_arm_flange"], dtype=np.float64)
        elif "position_xyz" in data and "orientation_xyzw" in data:
            position = np.asarray(data["position_xyz"], dtype=np.float64)
            quaternion = np.asarray(data["orientation_xyzw"], dtype=np.float64)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError("position_xyz must be (3,) and orientation_xyzw must be (4,)")
            frame_id = str(np.asarray(data["frame_id"]).item()) if "frame_id" in data else "r_base_link"
            return position, quaternion, frame_id
        else:
            raise ValueError("target NPZ requires T_robot_base_arm_flange or position_xyz/orientation_xyzw")
        frame_id = str(np.asarray(data["frame_id"]).item()) if "frame_id" in data else "r_base_link"
    position, quaternion = arm_flange_pose_xyzw(target)
    return position, quaternion, frame_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--topic", default="/right_arm/target_pose")
    parser.add_argument("--frame-id")
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--period-s", type=float, default=0.1)
    parser.add_argument("--execute", action="store_true", help="publish to the running Luban planner")
    args = parser.parse_args()
    if args.publish_count <= 0 or args.period_s <= 0:
        parser.error("--publish-count and --period-s must be positive")

    position, quaternion, frame_id = _load_target(args.target)
    frame_id = args.frame_id or frame_id
    print(f"AR5 target: frame={frame_id} position={np.array2string(position, precision=6)}")
    print(f"AR5 target: quaternion_xyzw={np.array2string(quaternion, precision=6)}")
    if not args.execute:
        print("Dry-run only. Add --execute to publish to Luban ROS2.")
        return

    import rclpy
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = rclpy.create_node("anydexretarget_ar5_target_publisher")
    publisher = node.create_publisher(PoseStamped, args.topic, 10)
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.pose.position.x, message.pose.position.y, message.pose.position.z = position.tolist()
    message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = quaternion.tolist()
    try:
        for _ in range(args.publish_count):
            message.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.0)
            node.get_logger().info(f"Published AR5 target to {args.topic}")
            time.sleep(args.period_s)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
