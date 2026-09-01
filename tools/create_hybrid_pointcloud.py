#!/usr/bin/env python3
"""Build a HUG-compatible cloud from measured RGB-D and mesh-completed surfaces.

Measured points are preserved verbatim.  Mesh points are added only when they
are farther than a merge radius from measured geometry, preventing a generated
mesh from replacing the camera's actual visible surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        if "points_camera" not in data or "colors_rgb" not in data:
            raise ValueError(f"{path} must contain points_camera and colors_rgb")
        points = np.asarray(data["points_camera"], dtype=np.float32)
        colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
        metadata = {
            key: np.asarray(data[key])
            for key in ("source_rgb", "source_depth", "source_mask", "intrinsics")
            if key in data
        }
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(f"Invalid cloud arrays in {path}: {points.shape}, {colors.shape}")
    if len(points) == 0 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError(f"{path} contains invalid camera-frame points")
    return points, colors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-pointcloud", type=Path, required=True,
                        help="Measured object cloud from the selected RGB-D mask.")
    parser.add_argument("--mesh-pointcloud", type=Path, required=True,
                        help="Photo-aligned full-surface mesh samples in the same camera frame.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-radius-mm", type=float, default=4.0,
                        help="Mesh points nearer than this to measured geometry are redundant.")
    parser.add_argument("--mesh-color", choices=("nearest-visible", "neutral"), default="nearest-visible",
                        help="Color convention for inferred mesh-only points.")
    args = parser.parse_args()
    if args.merge_radius_mm <= 0:
        raise ValueError("--merge-radius-mm must be positive")

    visible_points, visible_colors, visible_metadata = _load_cloud(args.visible_pointcloud)
    mesh_points, _mesh_colors, _ = _load_cloud(args.mesh_pointcloud)
    tree = cKDTree(visible_points)
    distances, nearest = tree.query(mesh_points, k=1)
    completion = distances > args.merge_radius_mm / 1000.0
    completion_points = mesh_points[completion]
    if args.mesh_color == "nearest-visible":
        completion_colors = visible_colors[nearest[completion]]
    else:
        completion_colors = np.full((len(completion_points), 3), 180, dtype=np.uint8)

    points = np.concatenate((visible_points, completion_points), axis=0)
    colors = np.concatenate((visible_colors, completion_colors), axis=0)
    is_observed = np.concatenate((
        np.ones(len(visible_points), dtype=np.uint8),
        np.zeros(len(completion_points), dtype=np.uint8),
    ))
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise RuntimeError("Hybrid output is not a valid camera-frame point cloud")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_camera=points.astype(np.float32),
        colors_rgb=colors.astype(np.uint8),
        is_observed=is_observed,
        source_type=np.asarray("hybrid_visible_rgbd_plus_mesh_completion"),
        visible_pointcloud=np.asarray(str(args.visible_pointcloud.resolve())),
        mesh_pointcloud=np.asarray(str(args.mesh_pointcloud.resolve())),
        merge_radius_m=np.asarray(args.merge_radius_mm / 1000.0, dtype=np.float32),
        **visible_metadata,
    )
    report = {
        "coordinate_frame": "anchor_rgbd_camera",
        "visible_pointcloud": str(args.visible_pointcloud.resolve()),
        "mesh_pointcloud": str(args.mesh_pointcloud.resolve()),
        "merge_radius_mm": args.merge_radius_mm,
        "visible_measured_points": int(len(visible_points)),
        "mesh_surface_points_input": int(len(mesh_points)),
        "mesh_points_redundant_with_visible_surface": int((~completion).sum()),
        "mesh_completion_points_added": int(completion.sum()),
        "hybrid_points_total": int(len(points)),
        "mesh_color_policy": args.mesh_color,
        "observed_flag": "is_observed=1 means measured RGB-D; 0 means mesh completion. Current pretrained HUG ignores this field.",
        "note": "HUG receives 4096 randomly sampled crop-local points from this cloud. The raw measured surface is not replaced by mesh points.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Hybrid HUG point cloud written")
    print(f"  measured visible points: {len(visible_points)}")
    print(f"  mesh completion points added: {int(completion.sum())}")
    print(f"  total: {len(points)}")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
