#!/usr/bin/env python3
"""Execute one prepared AnyDex grasp request through Luban ROS2.

The default ``preview`` stage never commands the arm or hand.  Hardware motion
requires both ``--execute`` and the explicit confirmation token.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anydexretarget.luban_arm import arm_flange_pose_xyzw, homogeneous_transform
from anydexretarget.luban_contract import (
    LUBAN_RIGHT_ARM_DONE_TOPIC,
    LUBAN_RIGHT_ARM_TARGET_TOPIC,
    LUBAN_RIGHT_HAND_CONTROLLER,
    LUBAN_RIGHT_HAND_DONE_TOPIC,
)


REQUIRED = {
    "base_frame",
    "T_robot_base_l25_hand_target",
    "T_robot_base_arm_flange_pregrasp",
    "T_robot_base_arm_flange_target",
    "l25_active_positions",
    "l25_active_joint_names",
}


def _read_request(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED - set(data.files)
        if missing:
            raise ValueError("grasp request missing: " + ", ".join(sorted(missing)))
        request = {key: np.asarray(data[key]).copy() for key in data.files}
    for key in ("T_robot_base_l25_hand_target", "T_robot_base_arm_flange_pregrasp", "T_robot_base_arm_flange_target"):
        homogeneous_transform(request[key], key)
    active = np.asarray(request["l25_active_positions"], dtype=np.float64)
    names = np.asarray(request["l25_active_joint_names"])
    if active.shape != (16,) or not np.isfinite(active).all() or names.shape != (16,):
        raise ValueError("L25 request must contain 16 finite active positions and names")
    return request


def _report(request: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    pregrasp, _ = arm_flange_pose_xyzw(request["T_robot_base_arm_flange_pregrasp"])
    target, _ = arm_flange_pose_xyzw(request["T_robot_base_arm_flange_target"])
    print(json.dumps({
        "planning_only": not args.execute,
        "stage": args.stage,
        "base_frame": str(request["base_frame"].item()),
        "candidate": str(request.get("candidate_id", np.asarray("")).item()),
        "pregrasp_position_m": pregrasp.tolist(),
        "grasp_position_m": target.tolist(),
        "l25_active_joint_names": request["l25_active_joint_names"].tolist(),
        "l25_active_positions_rad": request["l25_active_positions"].tolist(),
    }, indent=2))


def _wait_for_subscribers(rclpy, node, publisher, topic: str, timeout: float) -> None:
    """Wait for DDS discovery before sending a one-shot motion command."""
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if publisher.get_subscription_count() > 0:
            return
        rclpy.spin_once(node, timeout_sec=0.05)
    raise TimeoutError(f"No subscriber discovered on {topic}")


def _execute(request: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool, Float64MultiArray

    rclpy.init()
    node = rclpy.create_node("anydexretarget_luban_grasp_executor")
    wrist_pub = node.create_publisher(PoseStamped, args.wrist_topic, 10)
    arm_pub = node.create_publisher(PoseStamped, args.arm_topic, 10)
    hand_pub = node.create_publisher(Float64MultiArray, args.hand_topic, 10)
    done = {"arm": False, "hand": False}
    node.create_subscription(Bool, args.arm_done_topic, lambda msg: done.__setitem__("arm", bool(msg.data)), 10)
    node.create_subscription(Bool, args.hand_done_topic, lambda msg: done.__setitem__("hand", bool(msg.data)), 10)

    def pose(transform: np.ndarray) -> PoseStamped:
        position, quaternion = arm_flange_pose_xyzw(transform)
        message = PoseStamped()
        message.header.frame_id = str(request["base_frame"].item())
        message.header.stamp = node.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = position.tolist()
        message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = quaternion.tolist()
        return message

    def publish_arm(transform: np.ndarray, timeout: float, stage: str) -> None:
        done["arm"] = False
        _wait_for_subscribers(
            rclpy, node, arm_pub, args.arm_topic, args.discovery_timeout
        )
        for _ in range(args.publish_count):
            arm_pub.publish(pose(transform))
            rclpy.spin_once(node, timeout_sec=args.publish_period_s)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if done["arm"]:
                node.get_logger().info(
                    f"Arm stage '{stage}' completed via {args.arm_done_topic}"
                )
                return
        raise TimeoutError(f"No successful arm completion on {args.arm_done_topic}")

    def publish_hand(timeout: float) -> None:
        done["hand"] = False
        _wait_for_subscribers(
            rclpy, node, hand_pub, args.hand_topic, args.discovery_timeout
        )
        hand_pub.publish(Float64MultiArray(data=request["l25_active_positions"].tolist()))
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if done["hand"]:
                node.get_logger().info(
                    f"Hand stage 'close' completed via {args.hand_done_topic}"
                )
                return
        raise TimeoutError(f"No successful hand completion on {args.hand_done_topic}")

    try:
        if args.stage == "preview":
            _wait_for_subscribers(
                rclpy, node, wrist_pub, args.wrist_topic, args.discovery_timeout
            )
            for _ in range(args.publish_count):
                wrist_pub.publish(pose(request["T_robot_base_l25_hand_target"]))
                rclpy.spin_once(node, timeout_sec=args.publish_period_s)
            node.get_logger().info(
                f"Published diagnostic wrist goal to {args.wrist_topic}"
            )
            return
        wrist_pub.publish(pose(request["T_robot_base_l25_hand_target"]))
        node.get_logger().info(f"Published diagnostic wrist goal to {args.wrist_topic}")
        if args.stage in {"pregrasp", "all"}:
            publish_arm(
                request["T_robot_base_arm_flange_pregrasp"],
                args.arm_timeout,
                "pregrasp",
            )
        if args.stage in {"approach", "all"}:
            publish_arm(
                request["T_robot_base_arm_flange_target"],
                args.arm_timeout,
                "approach",
            )
        if args.stage in {"close", "all"}:
            publish_hand(args.hand_timeout)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--stage", choices=("preview", "pregrasp", "approach", "close", "all"), default="preview")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--wrist-topic", default="/anydex/right_l25_wrist_goal")
    parser.add_argument("--arm-topic", default=LUBAN_RIGHT_ARM_TARGET_TOPIC)
    parser.add_argument("--hand-topic", default=LUBAN_RIGHT_HAND_CONTROLLER)
    parser.add_argument("--arm-done-topic", default=LUBAN_RIGHT_ARM_DONE_TOPIC)
    parser.add_argument("--hand-done-topic", default=LUBAN_RIGHT_HAND_DONE_TOPIC)
    parser.add_argument("--arm-timeout", type=float, default=30.0)
    parser.add_argument("--hand-timeout", type=float, default=12.0)
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--publish-period-s", type=float, default=0.1)
    args = parser.parse_args()
    if (
        args.arm_timeout <= 0
        or args.hand_timeout <= 0
        or args.discovery_timeout <= 0
        or args.publish_count <= 0
        or args.publish_period_s <= 0
    ):
        parser.error("timeouts, publish-count and publish-period-s must be positive")
    if args.execute and args.confirm != "AR5_L25_CLEAR":
        parser.error("hardware or simulation execution requires --confirm AR5_L25_CLEAR")
    request = _read_request(args.request)
    _report(request, args)
    if not args.execute:
        print("Dry-run only. Add --execute --confirm AR5_L25_CLEAR to publish ROS commands.")
        return
    _execute(request, args)


if __name__ == "__main__":
    main()
