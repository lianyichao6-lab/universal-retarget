#!/usr/bin/env python3
"""Bridge the running ROS perception stack into AnyDex's RGB-D scene format.

The head ZED 2i + RobotPerception pipeline already publishes registered color,
FoundationStereo depth and an object mask as ROS topics. This command captures
one (or more) synchronized views and writes the exact files AnyDex expects, so
``tools/generate_hug_candidates.py`` can run without the Orbbec capture step or
the Hunyuan reconstruction backend:

    /clawbot_cam_head/left_color/image_raw        -> rgb.png
    /clawbot_cam_head/perception/stereo_depth     -> depth.png (uint16 mm)
    /clawbot_cam_head/left_color/camera_info      -> intrinsics.txt
    /clawbot_cam_head/perception/masks            -> mask.png + mask.json
    backproject(mask, depth, K)                   -> object_pointcloud.npz

Everything is expressed in the left-color optical frame (zed_left_camera_frame_optical),
which is what AnyDex treats as the anchor RGB-D camera frame.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

try:  # optional: capture the object 6D pose for downstream coordinate transforms
    from robot_perception_msgs.msg import LabeledPoseArray
except Exception:  # pragma: no cover - pose capture is best-effort
    LabeledPoseArray = None


class PerceptionCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros_perception_capture")
        self.args = args
        self.output = args.output
        self.bridge = CvBridge()
        self.camera_info: CameraInfo | None = None
        self.frame: tuple[np.ndarray, np.ndarray, np.ndarray, Image, Image] | None = None
        self.error: str | None = None
        self.saved = False
        self.cancelled = False
        self.finished = False
        self.view_count = args.start_index
        self.saved_views = 0
        self.last_save_at = 0.0
        self.started_at = time.monotonic()
        self.invalid_depth_frames = 0
        self.latest_pose_msg = None

        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        if args.pose_topic and LabeledPoseArray is not None:
            self.create_subscription(
                LabeledPoseArray, args.pose_topic, self._on_pose, qos_profile_sensor_data
            )
        color = message_filters.Subscriber(
            self, Image, args.color_topic, qos_profile=qos_profile_sensor_data
        )
        depth = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data
        )
        mask = message_filters.Subscriber(
            self, Image, args.mask_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color, depth, mask], queue_size=10, slop=args.sync_slop
        )
        self.sync.registerCallback(self._on_images)

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_pose(self, message) -> None:
        self.latest_pose_msg = message

    @staticmethod
    def _stamp(message: Image | CameraInfo) -> float:
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _on_images(self, color_msg: Image, depth_msg: Image, mask_msg: Image) -> None:
        if self.finished or self.cancelled or self.error or self.camera_info is None:
            return
        if time.monotonic() - self.started_at < self.args.warmup_seconds:
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
            if color.dtype != np.uint8 or color.shape[2] != 3:
                raise ValueError(f"Expected uint8 3-channel color, got {color.shape} {color.dtype}")
            if depth_m.dtype != np.float32 or depth_m.ndim != 2:
                raise ValueError(f"Expected float32 depth (meters), got {depth_m.shape} {depth_m.dtype}")
            if color.shape[:2] != depth_m.shape or color.shape[:2] != mask.shape:
                raise ValueError(
                    f"Color/depth/mask not registered: {color.shape[:2]} vs "
                    f"{depth_m.shape} vs {mask.shape}"
                )
            if (self.camera_info.width, self.camera_info.height) != (color.shape[1], color.shape[0]):
                raise ValueError("Color CameraInfo does not match color resolution")
            valid_ratio = float(np.count_nonzero(depth_m > 0)) / float(depth_m.size)
            if valid_ratio < self.args.min_valid_depth_ratio:
                self.invalid_depth_frames += 1
                if self.invalid_depth_frames == 1 or self.invalid_depth_frames % 30 == 0:
                    self.get_logger().warn(
                        "Ignoring frame with sparse stereo depth: "
                        f"valid={valid_ratio:.3%}, required={self.args.min_valid_depth_ratio:.3%}"
                    )
                return
            self.frame = (color, depth_m, mask, color_msg, depth_msg)
            if not self.args.preview:
                self.save()
        except Exception as exc:
            self.error = str(exc)
            self.get_logger().error(self.error)

    def _output_directory(self) -> Path:
        if not self.args.multi_view:
            return self.output
        return self.output / f"view_{self.view_count:03d}"

    @staticmethod
    def _foreground_point(mask: np.ndarray, depth_m: np.ndarray) -> tuple[int, int]:
        """A pixel on the object surface: median of masked, depth-valid pixels."""
        valid = (mask > 0) & (depth_m > 0)
        v, u = np.nonzero(valid)
        if u.size == 0:
            v, u = np.nonzero(mask > 0)
            if u.size == 0:
                raise RuntimeError("Object mask is empty")
        return int(np.median(u)), int(np.median(v))

    def _backproject(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_depth_m = self.args.max_depth_m
        valid = (mask > 0) & (depth_m > 0) & (depth_m < max_depth_m)
        v, u = np.nonzero(valid)
        if u.size == 0:
            raise ValueError("Object mask contains no valid depth points")
        z = depth_m[v, u]
        xyz = np.column_stack(
            (
                (u - K[0, 2]) * z / K[0, 0],
                (v - K[1, 2]) * z / K[1, 1],
                z,
            )
        ).astype(np.float32)
        colors = color[v, u].astype(np.uint8)
        pixels = np.column_stack((u, v)).astype(np.int32)
        if len(xyz) > self.args.max_points:
            indices = np.random.default_rng(self.args.seed).choice(
                len(xyz), size=self.args.max_points, replace=False
            )
            xyz, colors, pixels = xyz[indices], colors[indices], pixels[indices]
        return xyz, colors, pixels

    def _poses_to_dict(self) -> list[dict[str, object]] | None:
        if self.latest_pose_msg is None:
            return None
        return [
            {
                "label": lp.label,
                "position": [
                    lp.pose.position.x, lp.pose.position.y, lp.pose.position.z,
                ],
                "quaternion": [
                    lp.pose.orientation.x, lp.pose.orientation.y,
                    lp.pose.orientation.z, lp.pose.orientation.w,
                ],
                "frame_id": self.latest_pose_msg.header.frame_id,
            }
            for lp in self.latest_pose_msg.poses
        ]

    def save(self) -> None:
        if self.frame is None or self.camera_info is None:
            raise RuntimeError("No synchronized perception frame is available yet")
        if time.monotonic() - self.last_save_at < self.args.min_view_interval_seconds:
            self.get_logger().warn("Ignoring duplicate SPACE press; wait before saving next view")
            return
        color, depth_m, mask, color_msg, depth_msg = self.frame
        output = self._output_directory()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing capture: {output}. "
                "Use a new output path or a different --start-index."
            )
        output.mkdir(parents=True, exist_ok=True)

        K = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        if not cv2.imwrite(str(output / "rgb.png"), color):
            raise RuntimeError("Unable to write rgb.png")
        depth_mm = np.rint(depth_m * 1000.0)
        depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(output / "depth.png"), depth_mm):
            raise RuntimeError("Unable to write depth.png")
        np.savetxt(output / "intrinsics.txt", K, fmt="%.10f")
        if not cv2.imwrite(str(output / "mask.png"), mask):
            raise RuntimeError("Unable to write mask.png")

        point = (
            (int(self.args.point[0]), int(self.args.point[1]))
            if self.args.point is not None
            else self._foreground_point(mask, depth_m)
        )
        (output / "mask.json").write_text(
            json.dumps(
                {
                    "resolution": [int(color.shape[1]), int(color.shape[0])],
                    "foreground_point": list(point),
                    "mask_area_pixels": int(np.count_nonzero(mask > 0)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        xyz, colors, pixels = self._backproject(color, depth_m, mask, K)
        np.savez_compressed(
            output / "object_pointcloud.npz",
            points_camera=xyz,
            colors_rgb=colors,
            pixels_uv=pixels,
            intrinsics=K.astype(np.float32),
            source_rgb=np.asarray(str((output / "rgb.png").resolve())),
            source_depth=np.asarray(str((output / "depth.png").resolve())),
            source_mask=np.asarray(str((output / "mask.png").resolve())),
        )

        nonzero = depth_m[depth_m > 0]
        metadata = {
            "perception": "ZED 2i + RobotPerception (FoundationStereo + EfficientTAM)",
            "anchor_frame": color_msg.header.frame_id,
            "color_topic": self.args.color_topic,
            "depth_topic": self.args.depth_topic,
            "mask_topic": self.args.mask_topic,
            "camera_info_topic": self.args.camera_info_topic,
            "color_encoding": color_msg.encoding,
            "depth_encoding": depth_msg.encoding,
            "resolution": [int(color.shape[1]), int(color.shape[0])],
            "depth_nonzero_median_m": float(np.median(nonzero)) if nonzero.size else None,
            "depth_nonzero_pixels": int(nonzero.size),
            "foreground_point": list(point),
            "object_point_count": int(len(xyz)),
            "poses": self._poses_to_dict(),
        }
        (output / "capture_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        self.last_save_at = time.monotonic()
        self.saved_views += 1
        if self.args.multi_view:
            self.view_count += 1
        else:
            self.saved = True
            self.finished = True
        self.get_logger().info(
            f"Saved perception bridge capture to {output} "
            f"({color.shape[1]}x{color.shape[0]}, {len(xyz)} object points)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--color-topic", default="/clawbot_cam_head/left_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/clawbot_cam_head/left_color/camera_info")
    parser.add_argument("--depth-topic", default="/clawbot_cam_head/perception/stereo_depth")
    parser.add_argument("--mask-topic", default="/clawbot_cam_head/perception/masks")
    parser.add_argument(
        "--pose-topic",
        default="/clawbot_cam_head/perception/labeled_poses",
        help="Optional 6D pose topic captured into metadata (empty disables).",
    )
    parser.add_argument("--point", type=int, nargs=2, metavar=("U", "V"),
                        help="Foreground pixel override; default is the masked-depth median.")
    parser.add_argument("--preview", action="store_true", help="SPACE saves; Q or ESC cancels.")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Headless capture: ENTER saves, 'q' quits (no display needed).",
    )
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Keep preview open: every SPACE saves <output>/view_XXX.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--min-view-interval-seconds", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=20.0, help="0 means no timeout.")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--sync-slop", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.02)
    return parser.parse_args()


def _read_keyboard() -> str | None:
    """Non-blocking single-line read from stdin (works over SSH, no display)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return line.strip().lower() if line else None
    return None


def main() -> None:
    args = parse_args()
    if args.timeout < 0 or args.warmup_seconds < 0 or args.sync_slop < 0:
        raise ValueError("timeout, warmup-seconds and sync-slop cannot be negative")
    if args.max_depth_m <= 0 or args.max_points <= 0:
        raise ValueError("--max-depth-m and --max-points must be positive")
    if not 0 < args.min_valid_depth_ratio <= 1:
        raise ValueError("--min-valid-depth-ratio must be in (0, 1]")
    if args.multi_view and not (args.preview or args.keyboard):
        raise ValueError("--multi-view requires --preview or --keyboard so you control each saved view")
    if args.start_index < 0 or args.min_view_interval_seconds < 0:
        raise ValueError("--start-index and --min-view-interval-seconds cannot be negative")
    if args.pose_topic == "":
        args.pose_topic = None

    rclpy.init()
    node = PerceptionCapture(args)
    deadline = None if args.timeout == 0 else time.monotonic() + args.timeout
    window = "Perception bridge - SPACE save, Q/ESC finish"
    if args.keyboard:
        print("Headless capture: ENTER = save, 'q' = quit", flush=True)
    try:
        while (
            rclpy.ok()
            and not node.finished
            and not node.cancelled
            and node.error is None
            and (deadline is None or time.monotonic() < deadline)
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.keyboard and node.frame is not None:
                key = _read_keyboard()
                if key in ("", "s", " "):
                    node.save()
                elif key == "q":
                    if args.multi_view and node.saved_views:
                        node.finished = True
                    else:
                        node.cancelled = True
            elif args.preview and node.frame is not None:
                image = node.frame[0].copy()
                if args.multi_view:
                    instruction = (
                        f"View {node.view_count:03d}: SPACE save | Q / ESC finish"
                    )
                else:
                    instruction = "SPACE: save   Q / ESC: cancel"
                cv2.putText(
                    image, instruction, (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(window, image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    node.save()
                elif key in (ord("q"), 27):
                    if args.multi_view and node.saved_views:
                        node.finished = True
                    else:
                        node.cancelled = True
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    if node.error:
        raise RuntimeError(node.error)
    if node.cancelled:
        print("Capture cancelled; no files were written.")
    elif args.multi_view:
        print(f"Multi-view capture complete: {node.saved_views} views saved under {args.output}")
    elif not node.saved:
        raise TimeoutError(
            "Timed out waiting for synchronized color/depth/mask and CameraInfo "
            f"(ignored {node.invalid_depth_frames} sparse-depth frames)."
        )


if __name__ == "__main__":
    main()
