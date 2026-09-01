#!/usr/bin/env python3
"""Capture registered Orbbec RGB-D frames; preview mode saves on SPACE."""

from __future__ import annotations

import argparse
import json
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


class RgbdCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("orbbec_rgbd_capture")
        self.args = args
        self.bridge = CvBridge()
        self.output = args.output
        self.camera_info: CameraInfo | None = None
        self.frame: tuple[np.ndarray, np.ndarray, Image, Image] | None = None
        self.error: str | None = None
        self.saved = False
        self.cancelled = False
        self.finished = False
        self.view_count = args.start_index
        self.saved_views = 0
        self.last_save_at = 0.0
        self.started_at = time.monotonic()
        self.invalid_depth_frames = 0

        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        rgb = message_filters.Subscriber(
            self, Image, args.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        depth = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth], queue_size=10, slop=args.sync_slop
        )
        self.sync.registerCallback(self._on_images)

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    @staticmethod
    def _stamp(message: Image | CameraInfo) -> float:
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _on_images(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self.finished or self.cancelled or self.error or self.camera_info is None:
            return
        if time.monotonic() - self.started_at < self.args.warmup_seconds:
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if rgb.dtype != np.uint8 or rgb.shape != (rgb.shape[0], rgb.shape[1], 3):
                raise ValueError(f"Expected uint8 RGB image, got {rgb.shape} {rgb.dtype}")
            if depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError(f"Expected uint16 depth image, got {depth.shape} {depth.dtype}")
            if rgb.shape[:2] != depth.shape:
                raise ValueError(f"RGB/depth not registered: {rgb.shape[:2]} vs {depth.shape}")
            if (self.camera_info.width, self.camera_info.height) != (rgb.shape[1], rgb.shape[0]):
                raise ValueError("Color CameraInfo does not match RGB resolution")
            depth_mm = np.rint(depth.astype(np.float64) * self.args.depth_unit_mm)
            depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            valid_ratio = float(np.count_nonzero(depth_mm)) / float(depth_mm.size)
            if valid_ratio < self.args.min_valid_depth_ratio:
                self.invalid_depth_frames += 1
                if self.invalid_depth_frames == 1 or self.invalid_depth_frames % 30 == 0:
                    self.get_logger().warn(
                        "Ignoring invalid depth frame: "
                        f"valid={valid_ratio:.3%}, required={self.args.min_valid_depth_ratio:.3%}"
                    )
                return
            self.frame = (rgb, depth_mm, rgb_msg, depth_msg)
            if not self.args.preview:
                self.save()
        except Exception as exc:
            self.error = str(exc)
            self.get_logger().error(self.error)

    def _output_directory(self) -> Path:
        if not self.args.multi_view:
            return self.output
        return self.output / f"view_{self.view_count:03d}"

    def save(self) -> None:
        if self.frame is None or self.camera_info is None:
            raise RuntimeError("No synchronized RGB-D frame is available yet")
        if time.monotonic() - self.last_save_at < self.args.min_view_interval_seconds:
            self.get_logger().warn("Ignoring duplicate SPACE press; wait before saving next view")
            return
        rgb, depth_mm, rgb_msg, depth_msg = self.frame
        nonzero = depth_mm[depth_mm > 0]
        output = self._output_directory()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing capture: {output}. "
                "Use a new output path or a different --start-index."
            )
        output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Unable to write rgb.png")
        if not cv2.imwrite(str(output / "depth.png"), depth_mm):
            raise RuntimeError("Unable to write depth.png")
        intrinsics = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        np.savetxt(output / "intrinsics.txt", intrinsics, fmt="%.10f")
        metadata = {
            "camera": "Orbbec Gemini 335 via ROS2",
            "multi_view_session": self.args.multi_view,
            "view_index": self.view_count if self.args.multi_view else None,
            "rgb_topic": self.args.rgb_topic,
            "depth_topic": self.args.depth_topic,
            "camera_info_topic": self.args.camera_info_topic,
            "rgb_encoding": rgb_msg.encoding,
            "depth_encoding": depth_msg.encoding,
            "rgb_timestamp_s": self._stamp(rgb_msg),
            "depth_timestamp_s": self._stamp(depth_msg),
            "camera_info_timestamp_s": self._stamp(self.camera_info),
            "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "depth_unit_mm_multiplier": self.args.depth_unit_mm,
            "depth_nonzero_median_mm": float(np.median(nonzero)),
            "depth_nonzero_pixels": int(nonzero.size),
            "intrinsics": intrinsics.tolist(),
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
            f"Saved registered RGB-D capture to {output} "
            f"({rgb.shape[1]}x{rgb.shape[0]}, median depth {np.median(nonzero):.1f} mm)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-topic", default="/scene_camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/scene_camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/scene_camera/color/camera_info")
    parser.add_argument("--preview", action="store_true", help="SPACE saves; Q or ESC cancels.")
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help=(
            "Keep preview open: every SPACE saves <output>/view_XXX; "
            "Q or ESC finishes the session."
        ),
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First view index in --multi-view mode (default: 0).",
    )
    parser.add_argument(
        "--min-view-interval-seconds", type=float, default=0.7,
        help="Minimum delay between saved views in --multi-view mode.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="0 means no timeout.")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--depth-unit-mm", type=float, default=1.0)
    parser.add_argument(
        "--min-valid-depth-ratio", type=float, default=0.02,
        help="Ignore frames with fewer valid depth pixels than this ratio (default: 0.02).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.timeout < 0
        or args.warmup_seconds < 0
        or args.depth_unit_mm <= 0
        or not 0 < args.min_valid_depth_ratio <= 1
    ):
        raise ValueError(
            "timeout cannot be negative; depth-unit-mm must be positive; "
            "min-valid-depth-ratio must be in (0, 1]"
        )
    if args.multi_view and not args.preview:
        raise ValueError("--multi-view requires --preview so you control each saved view")
    if args.start_index < 0 or args.min_view_interval_seconds < 0:
        raise ValueError("--start-index and --min-view-interval-seconds cannot be negative")
    rclpy.init()
    node = RgbdCapture(args)
    deadline = None if args.timeout == 0 else time.monotonic() + args.timeout
    window = "Gemini 335 - SPACE save, Q/ESC finish"
    try:
        while (
            rclpy.ok()
            and not node.finished
            and not node.cancelled
            and node.error is None
            and (deadline is None or time.monotonic() < deadline)
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.preview and node.frame is not None:
                image = cv2.cvtColor(node.frame[0], cv2.COLOR_RGB2BGR)
                if args.multi_view:
                    instruction = (
                        f"View {node.view_count:03d}: SPACE save | rotate object | "
                        "Q / ESC finish"
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
        print(
            f"Multi-view capture complete: {node.saved_views} views saved under {args.output}"
        )
    elif not node.saved:
        raise TimeoutError(
            "Timed out waiting for usable registered RGB-D and CameraInfo "
            f"(ignored {node.invalid_depth_frames} invalid depth frames)."
        )


if __name__ == "__main__":
    main()
