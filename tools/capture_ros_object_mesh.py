#!/usr/bin/env python3
"""Bridge the running perception's object mesh into AnyDex's collision mesh.

The RobotPerception pipeline publishes ``<ns>/perception/mesh_cloud/obj_<tag>``
as a PointCloud2 in the ``world`` frame: the object's CAD surface sampled at the
estimated 6D pose. AnyDex's collision-aware planning needs a triangle mesh in
the anchor camera optical frame (the same frame as the bridged
``object_pointcloud.npz`` and the HUG candidates).

This tool captures that cloud once, transforms ``world -> camera`` with
``T_world_headcam`` from the calibration file, reconstructs a closed triangle
mesh (convex hull by default), and writes ``<output>.ply`` plus the raw camera
points for verification.

Run with the same conda env the perception uses, after sourcing ROS:
    source /opt/ros/jazzy/setup.bash
    ~/miniconda3/envs/robot_perception/bin/python tools/capture_ros_object_mesh.py \
      --output outputs/perception_bridge/run_001/object_mesh_camera.ply
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
import trimesh
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


def _read_cloud_points(message: PointCloud2) -> np.ndarray:
    """Parse an XYZ PointCloud2 into an (N, 3) float64 array."""
    from sensor_msgs_py import point_cloud2

    pts = point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
    array = np.array([tuple(p) for p in pts], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"Invalid point cloud: {array.shape}")
    return array


def _find_mesh_cloud_topic(node: Node) -> str | None:
    """Locate the first ``*/perception/mesh_cloud/obj_*`` topic, if any."""
    for name, _types in node.get_topic_names_and_types():
        if "/perception/mesh_cloud/obj_" in name:
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-topic", default="",
                        help="Explicit mesh_cloud topic; auto-discovered if empty.")
    parser.add_argument("--calib-file",
                        default="/botclaw/calib_results/full_calibration_result.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Seconds to wait for the point cloud (0 = no timeout).")
    parser.add_argument("--method", choices=("convex_hull",), default="convex_hull")
    args = parser.parse_args()

    rclpy.init()
    node = Node("object_mesh_capture")
    latest: list[PointCloud2] = []

    def _on_cloud(message: PointCloud2) -> None:
        latest.append(message)

    topic = args.cloud_topic or _find_mesh_cloud_topic(node)
    if topic is None:
        raise RuntimeError("No mesh_cloud topic found; pass --cloud-topic explicitly.")
    node.create_subscription(PointCloud2, topic, _on_cloud, qos_profile_sensor_data)
    node.get_logger().info(f"Waiting for point cloud on {topic}...")

    deadline = None if args.timeout == 0 else time.monotonic() + args.timeout
    while not latest and rclpy.ok() and (deadline is None or time.monotonic() < deadline):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not latest:
        node.destroy_node()
        rclpy.shutdown()
        raise TimeoutError(f"No point cloud received on {topic} within {args.timeout}s")

    message = latest[-1]
    calib = np.load(args.calib_file, allow_pickle=True)
    if "T_world_headcam" not in calib and "T_world_cam" not in calib:
        raise KeyError(f"{args.calib_file} has neither T_world_headcam nor T_world_cam")
    T_world_cam = np.asarray(
        calib["T_world_headcam"] if "T_world_headcam" in calib else calib["T_world_cam"],
        dtype=np.float64,
    )
    T_cam_world = np.linalg.inv(T_world_cam)

    pts_world = _read_cloud_points(message)
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

    if np.any(pts_cam[:, 2] <= 0):
        node.get_logger().warn(
            f"{np.count_nonzero(pts_cam[:, 2] <= 0)}/{len(pts_cam)} points behind camera "
            "(z<=0); check the T_world_headcam frame convention."
        )

    if args.method == "convex_hull":
        mesh = trimesh.convex.convex_hull(pts_cam)
    else:  # pragma: no cover
        raise ValueError(f"Unknown method {args.method}")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output))
    points_path = output.with_suffix(".npz")
    np.savez_compressed(points_path, points_camera=pts_cam, points_world=pts_world)
    metadata = {
        "cloud_topic": topic,
        "source_frame": message.header.frame_id,
        "target_frame": "zed_left_camera_frame_optical (anchor camera)",
        "point_count": int(len(pts_cam)),
        "reconstruction": args.method,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "bounds_camera_m": {
            "min": mesh.vertices.min(axis=0).tolist(),
            "max": mesh.vertices.max(axis=0).tolist(),
        },
        "calib_file": str(Path(args.calib_file).resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    node.get_logger().info(
        f"Exported {args.method} mesh ({len(mesh.vertices)} verts, {len(mesh.faces)} faces) "
        f"to {output} ({len(pts_cam)} points, frame={message.header.frame_id} -> camera)"
    )
    print(f"mesh:      {output}")
    print(f"points:    {points_path}")
    print(f"metadata:  {output.with_suffix('.json')}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
