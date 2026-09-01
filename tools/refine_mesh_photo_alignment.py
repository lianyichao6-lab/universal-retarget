#!/usr/bin/env python3
"""Refine an anchor-camera mesh to the selected object silhouette in one RGB image.

This applies a small similarity adjustment to a mesh already in camera
coordinates.  The selected mask component anchors its image position and
apparent size; RGB-D remains the source of metric scale in the preceding ICP
step.  It is a calibration refinement, not a 3-D reconstruction method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [item for item in raw.geometry.values() if isinstance(item, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    return raw


def _component_at(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    x, y = point
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not mask[y, x]:
        raise ValueError("--point must be inside the selected object mask")
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    label = int(labels[y, x])
    if label == 0 or count <= 1:
        raise ValueError("No foreground mask component at --point")
    return labels == label


def _project_mask(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    depth = vertices[:, 2]
    pixels = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-6
    pixels[valid, 0] = K[0, 0] * vertices[valid, 0] / depth[valid] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * vertices[valid, 1] / depth[valid] + K[1, 2]
    result = np.zeros((height, width), dtype=np.uint8)
    for face in faces[valid[faces].all(axis=1)]:
        polygon = np.rint(pixels[face]).astype(np.int32)
        if ((polygon[:, 0] < -1).all() or (polygon[:, 0] >= width).all() or
                (polygon[:, 1] < -1).all() or (polygon[:, 1] >= height).all()):
            continue
        cv2.fillConvexPoly(result, polygon, 255, lineType=cv2.LINE_8)
    return result > 0


def _box(mask: np.ndarray) -> tuple[float, float, float, float, int]:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise RuntimeError("Projected mesh does not overlap the camera image")
    left, right = float(x.min()), float(x.max())
    top, bottom = float(y.min()), float(y.max())
    return left, top, right, bottom, int(len(x))


def _box_dict(box: tuple[float, float, float, float, int]) -> dict[str, float | int]:
    left, top, right, bottom, area = box
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "width": right - left + 1.0, "height": bottom - top + 1.0,
        "center_u": (left + right) / 2.0, "center_v": (top + bottom) / 2.0,
        "area_pixels": area,
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Initial mesh already in anchor camera coordinates.")
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--point", type=int, nargs=2, metavar=("U", "V"), required=True,
                        help="Foreground click used to select one connected mask component.")
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--output-pointcloud", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max-scale-step", type=float, default=1.35,
                        help="Reject implausibly large per-iteration apparent-size correction.")
    args = parser.parse_args()
    if args.samples < 1000 or args.iterations < 1 or args.max_scale_step <= 1.0:
        raise ValueError("Invalid --samples, --iterations, or --max-scale-step")

    mask_image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask}")
    target = _component_at(mask_image > 0, tuple(args.point))
    K = np.loadtxt(args.intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 K matrix, got {K.shape}")
    mesh = _load_mesh(args.mesh).copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    target_box = _box(target)
    before_mask = _project_mask(vertices, faces, K, target.shape)
    before_box = _box(before_mask)
    applied_steps: list[dict[str, float]] = []

    for _ in range(args.iterations):
        projected = _project_mask(vertices, faces, K, target.shape)
        source_box = _box(projected)
        source = _box_dict(source_box)
        desired = _box_dict(target_box)
        width_ratio = float(desired["width"] / source["width"])
        height_ratio = float(desired["height"] / source["height"])
        size_ratio = float(np.sqrt(width_ratio * height_ratio))
        size_ratio = float(np.clip(size_ratio, 1.0 / args.max_scale_step, args.max_scale_step))
        center = vertices.mean(axis=0)
        vertices = center + (vertices - center) * size_ratio
        # Shift the camera-space centroid so the projected bbox centers meet.
        z = float(max(center[2], 1e-4))
        du = float(desired["center_u"] - source["center_u"])
        dv = float(desired["center_v"] - source["center_v"])
        delta = np.array((du * z / K[0, 0], dv * z / K[1, 1], 0.0), dtype=np.float64)
        vertices += delta
        applied_steps.append({"apparent_scale": size_ratio, "delta_x_m": float(delta[0]), "delta_y_m": float(delta[1])})
        if abs(size_ratio - 1.0) < 0.002 and abs(du) < 0.5 and abs(dv) < 0.5:
            break

    mesh.vertices = vertices
    after_mask = _project_mask(vertices, faces, K, target.shape)
    after_box = _box(after_mask)
    if np.any(vertices[:, 2] <= 0):
        raise RuntimeError("Refinement put mesh behind the camera; rejecting output")
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_mesh)
    np.random.seed(0)
    points, _ = trimesh.sample.sample_surface(mesh, args.samples)
    args.output_pointcloud.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_pointcloud,
        points_camera=points.astype(np.float32),
        colors_rgb=np.full((len(points), 3), 180, dtype=np.uint8),
        source_type=np.asarray("hunyuan_mesh_photo_refined_anchor_camera"),
        source_mesh=np.asarray(str(args.output_mesh.resolve())),
    )
    report = {
        "coordinate_frame": "anchor_rgbd_camera",
        "initial_mesh": str(args.mesh.resolve()),
        "mask": str(args.mask.resolve()),
        "selected_foreground_point": list(args.point),
        "target_mask_bbox": _box_dict(target_box),
        "mesh_bbox_before": _box_dict(before_box),
        "mesh_bbox_after": _box_dict(after_box),
        "silhouette_iou_before": _iou(before_mask, target),
        "silhouette_iou_after": _iou(after_mask, target),
        "steps": applied_steps,
        "note": "Photo silhouette refinement fixes apparent 2-D pose/size only. Inspect the overlay; it does not prove hidden-surface geometry or physical grasp stability.",
    }
    report_path = args.output_pointcloud.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"photo-refined mesh: {args.output_mesh}")
    print(f"HUG point cloud: {args.output_pointcloud}")
    print(f"silhouette IoU: {report['silhouette_iou_before']:.3f} -> {report['silhouette_iou_after']:.3f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
