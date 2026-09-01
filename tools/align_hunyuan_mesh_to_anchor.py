#!/usr/bin/env python3
"""Align a Hunyuan canonical mesh to an anchor RGB-D object point cloud.

The alignment is an automatic initial estimate: PCA supplies 24 proper
orientation hypotheses, a metric scale estimate, and trimmed ICP refines the
rigid pose against the visible RGB-D surface.  The output mesh and sampled
point cloud are in the anchor camera frame used by HUG.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Canonical Hunyuan GLB/OBJ/PLY mesh.")
    parser.add_argument("--anchor-pointcloud", type=Path, required=True, help="Anchor object_pointcloud.npz.")
    parser.add_argument("--output-mesh", type=Path, required=True, help="Aligned mesh PLY/GLB output.")
    parser.add_argument("--output-pointcloud", type=Path, required=True, help="HUG-compatible aligned surface cloud (.npz).")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--alignment-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--trim-fraction", type=float, default=0.30)
    parser.add_argument("--max-correspondence-m", type=float, default=0.030)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [item for item in raw.geometry.values() if isinstance(item, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    components = raw.split(only_watertight=False)
    mesh = max(components, key=lambda item: len(item.faces)).copy()
    if len(mesh.faces) < 100:
        raise ValueError("Largest mesh component is unexpectedly small")
    return mesh


def load_anchor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "points_camera" not in data:
            raise ValueError(f"Missing points_camera: {path}")
        points = np.asarray(data["points_camera"], dtype=np.float64)
        colors = np.asarray(data.get("colors_rgb", np.full((len(points), 3), 180)), dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError(f"Invalid anchor point cloud: {points.shape}")
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Anchor cloud must contain finite positive camera-Z points")
    return points, colors


def pca_axes(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def axis_extents(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    projected = (points - points.mean(axis=0)) @ axes
    return np.percentile(projected, 97.5, axis=0) - np.percentile(projected, 2.5, axis=0)


def signed_permutations() -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base * np.asarray(signs)[None, :]
            if np.linalg.det(candidate) > 0:
                result.append(candidate)
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, int]:
    tree = cKDTree(target)
    transform = initial.copy()
    for _ in range(args.icp_iterations):
        moved = transform_points(source, transform)
        distances, indices = tree.query(moved, k=1)
        valid = distances <= args.max_correspondence_m
        if valid.sum() < 32:
            break
        threshold = np.quantile(distances[valid], args.trim_fraction)
        valid &= distances <= threshold
        if valid.sum() < 32:
            break
        delta = kabsch(moved[valid], target[indices[valid]])
        transform = delta @ transform
    moved = transform_points(source, transform)
    distances, _ = tree.query(moved, k=1)
    valid = distances <= args.max_correspondence_m
    if valid.sum() >= 32:
        valid &= distances <= np.quantile(distances[valid], args.trim_fraction)
    rmse = float(np.sqrt(np.mean(distances[valid] ** 2))) if valid.any() else float("inf")
    return transform, rmse, int(valid.sum())


def main() -> None:
    args = parse_args()
    if args.samples < 1000 or args.alignment_samples < 1000 or args.icp_iterations <= 0:
        raise ValueError("--samples/--alignment-samples must be >= 1000 and --icp-iterations must be positive")
    if not 0.05 <= args.trim_fraction <= 1.0 or args.max_correspondence_m <= 0:
        raise ValueError("Invalid ICP trimming or correspondence distance")
    mesh = load_mesh(args.mesh)
    target, _ = load_anchor(args.anchor_pointcloud)
    np.random.seed(args.seed)
    source, _ = trimesh.sample.sample_surface(mesh, args.alignment_samples)
    source_axes = pca_axes(source)
    target_axes = pca_axes(target)
    source_extent = axis_extents(source, source_axes)
    target_extent = axis_extents(target, target_axes)

    best: tuple[np.ndarray, float, int, float, int] | None = None
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    for index, signed in enumerate(signed_permutations()):
        rotation = target_axes @ signed @ source_axes.T
        mapped_extent = source_extent[np.argmax(np.abs(signed), axis=1)]
        scale = float(np.median(target_extent / np.maximum(mapped_extent, 1e-9)))
        initial = np.eye(4)
        initial[:3, :3] = scale * rotation
        initial[:3, 3] = target_center - scale * rotation @ source_center
        refined, rmse, correspondences = icp(source, target, initial, args)
        candidate = (refined, rmse, correspondences, scale, index)
        if best is None or (rmse, -correspondences) < (best[1], -best[2]):
            best = candidate
    if best is None or not np.isfinite(best[1]):
        raise RuntimeError("No valid alignment found; inspect the mesh and anchor segmentation")
    transform, rmse, correspondences, initial_scale, hypothesis = best

    vertices = transform_points(np.asarray(mesh.vertices, dtype=np.float64), transform)
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = vertices
    if np.any(vertices[:, 2] <= 0):
        raise RuntimeError("Aligned mesh has non-positive camera Z; reject this automatic alignment")
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    aligned_mesh.export(args.output_mesh)
    points, face_indices = trimesh.sample.sample_surface(aligned_mesh, args.samples)
    colors = np.full((len(points), 3), 180, dtype=np.uint8)
    args.output_pointcloud.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_pointcloud,
        points_camera=points.astype(np.float32),
        colors_rgb=colors,
        source_type=np.asarray("hunyuan3d_mesh_aligned_to_anchor"),
        source_mesh=np.asarray(str(args.mesh.resolve())),
        model_to_camera=transform.astype(np.float64),
    )
    report = {
        "source_mesh": str(args.mesh.resolve()),
        "anchor_pointcloud": str(args.anchor_pointcloud.resolve()),
        "coordinate_frame": "anchor_rgbd_camera",
        "largest_component_faces": int(len(mesh.faces)),
        "largest_component_watertight": bool(mesh.is_watertight),
        "pca_hypothesis": hypothesis,
        "initial_uniform_scale": initial_scale,
        "model_to_camera": transform.tolist(),
        "trimmed_icp_rmse_m": rmse,
        "trimmed_icp_correspondences": correspondences,
        "mesh_vertices": int(len(aligned_mesh.vertices)),
        "mesh_faces": int(len(aligned_mesh.faces)),
        "note": "Automatic visible-surface alignment. Inspect overlay before using this point cloud for HUG.",
    }
    report_path = args.output_pointcloud.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"aligned mesh: {args.output_mesh}")
    print(f"HUG point cloud: {args.output_pointcloud}")
    print(f"trimmed ICP: rmse={rmse:.5f} m, correspondences={correspondences}, scale={initial_scale:.6f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
