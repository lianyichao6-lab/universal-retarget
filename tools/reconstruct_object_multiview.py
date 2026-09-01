#!/usr/bin/env python3
"""Register and fuse masked RGB-D object point clouds into an object model.

Each input cloud must come from ``object_mask_to_pointcloud.py``.  The first
cloud is the anchor coordinate frame, which keeps the result compatible with
``generate_hug_candidates.py``: use that first view's RGB, depth, intrinsics,
and mask when generating HUG samples.  Additional views are registered using
PCA hypotheses followed by trimmed point-to-point ICP.

This is a surface reconstruction, not shape completion.  It only contains
surfaces seen by at least one captured view.  The optional mesh is a voxel
surface proxy for inspection; it must not be treated as a collision-accurate,
watertight CAD model.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ObjectCloud:
    path: Path
    points: np.ndarray
    colors: np.ndarray
    pixels: np.ndarray
    intrinsics: np.ndarray
    source_mask: Path | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointcloud",
        type=Path,
        action="append",
        required=True,
        help="Input .npz from object_mask_to_pointcloud.py; first one is the anchor.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Fused .npz output.")
    parser.add_argument("--ply", type=Path, help="Fused point-cloud PLY output.")
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Optional voxel surface-proxy mesh PLY for inspection only.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument(
        "--max-depth-deviation-m",
        type=float,
        default=0.08,
        help=(
            "Reject input points whose camera-frame depth differs from that view's "
            "median by more than this amount. Set 0 to disable (default: 0.08)."
        ),
    )
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--max-correspondence-m", type=float, default=0.025)
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.80,
        help="Fraction of nearest correspondences retained by each ICP step.",
    )
    parser.add_argument(
        "--min-correspondences",
        type=int,
        default=100,
        help="Minimum retained correspondences needed to accept an ICP step.",
    )
    parser.add_argument(
        "--max-registration-rmse-m",
        type=float,
        default=0.012,
        help="Views above this post-registration RMSE are retained but flagged.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.pointcloud) < 2:
        parser.error("At least two --pointcloud inputs are required for multi-view fusion")
    if args.voxel_size <= 0 or args.max_correspondence_m <= 0:
        parser.error("--voxel-size and --max-correspondence-m must be positive")
    if args.max_depth_deviation_m < 0:
        parser.error("--max-depth-deviation-m cannot be negative")
    if args.icp_iterations <= 0 or args.min_correspondences < 3:
        parser.error("--icp-iterations must be positive and --min-correspondences >= 3")
    if not 0 < args.trim_fraction <= 1:
        parser.error("--trim-fraction must be in (0, 1]")
    if args.max_registration_rmse_m <= 0:
        parser.error("--max-registration-rmse-m must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output exists: {args.output}; pass --overwrite to replace it")
    return args


def _load_cloud(path: Path) -> ObjectCloud:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("points_camera", "colors_rgb", "pixels_uv", "intrinsics")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is not an object point cloud; missing {missing}")
        points = np.asarray(data["points_camera"], dtype=np.float64)
        colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
        pixels = np.asarray(data["pixels_uv"], dtype=np.int32)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        source_mask = (
            Path(str(data["source_mask"].item())) if "source_mask" in data else None
        )
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise ValueError(f"Invalid points_camera shape in {path}: {points.shape}")
    if colors.shape != (len(points), 3) or pixels.shape != (len(points), 2):
        raise ValueError(f"Point attributes do not match points_camera in {path}")
    if intrinsics.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid intrinsics or non-finite points in {path}")
    return ObjectCloud(path, points, colors, pixels, intrinsics, source_mask)


def _filter_depth_outliers(
    cloud: ObjectCloud, max_depth_deviation_m: float
) -> tuple[ObjectCloud, int]:
    """Drop isolated background depth from a masked single-view cloud."""
    if max_depth_deviation_m == 0:
        return cloud, 0
    median_depth = float(np.median(cloud.points[:, 2]))
    keep = np.abs(cloud.points[:, 2] - median_depth) <= max_depth_deviation_m
    removed = int(np.count_nonzero(~keep))
    if int(np.count_nonzero(keep)) < 32:
        raise ValueError(
            f"Depth outlier filter left fewer than 32 points for {cloud.path}; "
            "increase --max-depth-deviation-m or redraw the mask."
        )
    return (
        ObjectCloud(
            cloud.path, cloud.points[keep], cloud.colors[keep], cloud.pixels[keep],
            cloud.intrinsics, cloud.source_mask
        ),
        removed,
    )


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the transform that applies right first, then left."""
    return left @ right


def _voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    counts = np.bincount(inverse)
    summed_points = np.zeros((len(first), 3), dtype=np.float64)
    summed_colors = np.zeros((len(first), 3), dtype=np.float64)
    np.add.at(summed_points, inverse, points)
    np.add.at(summed_colors, inverse, colors.astype(np.float64))
    return summed_points / counts[:, None], np.rint(summed_colors / counts[:, None]).astype(np.uint8)


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _proper_signed_permutations() -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signed = base * np.asarray(signs)[None, :]
            if np.linalg.det(signed) > 0:
                result.append(signed)
    return result


def _pca_hypotheses(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    source_axes = _pca_axes(source)
    target_axes = _pca_axes(target)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    hypotheses: list[np.ndarray] = []
    for signed in _proper_signed_permutations():
        rotation = target_axes @ signed @ source_axes.T
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_center - rotation @ source_center
        hypotheses.append(transform)
    return hypotheses


@dataclass(frozen=True)
class Registration:
    transform: np.ndarray
    rmse_m: float
    correspondences: int
    iterations: int
    pca_hypothesis: int


def _icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    max_distance: float,
    trim_fraction: float,
    min_correspondences: int,
    iterations: int,
) -> tuple[np.ndarray, float, int, int]:
    tree = cKDTree(target)
    transform = initial.copy()
    previous_rmse = np.inf
    retained = 0
    for step in range(iterations):
        transformed = _apply(source, transform)
        distances, indices = tree.query(transformed, k=1)
        valid = distances <= max_distance
        if valid.sum() < min_correspondences:
            break
        cutoff = np.quantile(distances[valid], trim_fraction)
        valid &= distances <= cutoff
        retained = int(valid.sum())
        if retained < min_correspondences:
            break
        delta = _kabsch(transformed[valid], target[indices[valid]])
        transform = _compose(delta, transform)
        rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
        if abs(previous_rmse - rmse) < 1e-6:
            return transform, rmse, retained, step + 1
        previous_rmse = rmse

    transformed = _apply(source, transform)
    distances, _ = tree.query(transformed, k=1)
    valid = distances <= max_distance
    if valid.sum() >= min_correspondences:
        cutoff = np.quantile(distances[valid], trim_fraction)
        valid &= distances <= cutoff
    retained = int(valid.sum())
    rmse = float(np.sqrt(np.mean(distances[valid] ** 2))) if retained else np.inf
    return transform, rmse, retained, iterations


def _register(
    source: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> Registration:
    best: Registration | None = None
    for index, initial in enumerate(_pca_hypotheses(source, target)):
        transform, rmse, count, steps = _icp(
            source,
            target,
            initial,
            args.max_correspondence_m,
            args.trim_fraction,
            args.min_correspondences,
            args.icp_iterations,
        )
        candidate = Registration(transform, rmse, count, steps, index)
        if best is None or (candidate.rmse_m, -candidate.correspondences) < (
            best.rmse_m,
            -best.correspondences,
        ):
            best = candidate
    if best is None or not np.isfinite(best.rmse_m):
        raise RuntimeError("ICP found no valid registration; capture more overlap between views")
    return best


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_surface_proxy(path: Path, points: np.ndarray, voxel_size: float) -> int:
    try:
        from trimesh.voxel.ops import points_to_marching_cubes
    except ImportError as exc:
        raise RuntimeError("trimesh is required for --mesh") from exc
    mesh = points_to_marching_cubes(points, pitch=voxel_size)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Unable to create surface proxy mesh from fused points")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return int(len(mesh.faces))


def _project_pixels(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-8)
    u = np.rint(intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2])
    v = np.rint(intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2])
    return np.column_stack((u, v)).astype(np.int32)


def main() -> None:
    args = _parse_args()
    loaded_clouds = [_load_cloud(path) for path in args.pointcloud]
    filtered = [
        _filter_depth_outliers(cloud, args.max_depth_deviation_m)
        for cloud in loaded_clouds
    ]
    clouds = [cloud for cloud, _ in filtered]
    depth_outliers_removed = [removed for _, removed in filtered]
    anchor = clouds[0]
    anchor_points, anchor_colors = _voxel_downsample(
        anchor.points, anchor.colors, args.voxel_size
    )
    fused_points = anchor_points
    fused_colors = anchor_colors
    transforms = [np.eye(4, dtype=np.float64)]
    registrations: list[dict[str, object]] = [
        {
            "view": 0,
            "pointcloud": str(anchor.path.resolve()),
            "status": "anchor",
            "rmse_m": 0.0,
            "correspondences": int(len(anchor_points)),
            "input_points": int(len(loaded_clouds[0].points)),
            "depth_outliers_removed": depth_outliers_removed[0],
            "transform_to_anchor": transforms[0].tolist(),
        }
    ]

    for view_index, cloud in enumerate(clouds[1:], start=1):
        source_points, source_colors = _voxel_downsample(
            cloud.points, cloud.colors, args.voxel_size
        )
        registration = _register(source_points, fused_points, args)
        aligned_points = _apply(source_points, registration.transform)
        fused_points = np.concatenate((fused_points, aligned_points), axis=0)
        fused_colors = np.concatenate((fused_colors, source_colors), axis=0)
        fused_points, fused_colors = _voxel_downsample(
            fused_points, fused_colors, args.voxel_size
        )
        transforms.append(registration.transform)
        status = (
            "accepted"
            if registration.rmse_m <= args.max_registration_rmse_m
            else "warning_high_rmse"
        )
        registrations.append(
            {
                "view": view_index,
                "pointcloud": str(cloud.path.resolve()),
                "status": status,
                "input_points": int(len(loaded_clouds[view_index].points)),
                "depth_outliers_removed": depth_outliers_removed[view_index],
                "rmse_m": registration.rmse_m,
                "correspondences": registration.correspondences,
                "icp_iterations": registration.iterations,
                "pca_hypothesis": registration.pca_hypothesis,
                "transform_to_anchor": registration.transform.tolist(),
            }
        )
        print(
            f"view {view_index}: {status}, rmse={registration.rmse_m * 1000:.2f} mm, "
            f"correspondences={registration.correspondences}, "
            f"depth_outliers_removed={depth_outliers_removed[view_index]}"
        )

    # The fused model uses the first capture's camera frame.  Its original
    # source pixels remain meaningful for HUG's selected target-point check;
    # newly revealed points use their projection into that anchor camera.
    anchor_pixels = _project_pixels(fused_points, anchor.intrinsics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_camera=fused_points.astype(np.float32),
        colors_rgb=fused_colors.astype(np.uint8),
        pixels_uv=anchor_pixels,
        intrinsics=anchor.intrinsics.astype(np.float32),
        source_mask=np.asarray(str(anchor.source_mask.resolve()))
        if anchor.source_mask is not None
        else np.asarray(""),
        source_rgb=np.asarray(str(anchor.path.resolve())),
        reconstruction_type=np.asarray("multiview_rgbd_surface_fusion"),
        coordinate_frame=np.asarray("anchor_view_camera"),
    )
    ply_path = args.ply or args.output.with_suffix(".ply")
    _write_ply(ply_path, fused_points, fused_colors)
    mesh_faces = None
    if args.mesh is not None:
        mesh_faces = _write_surface_proxy(args.mesh, fused_points, args.voxel_size)
    warnings = [
        f"view {row['view']} registration RMSE exceeds threshold"
        for row in registrations
        if row["status"] == "warning_high_rmse"
    ]
    metadata = {
        "reconstruction_type": "multiview_rgbd_surface_fusion",
        "coordinate_frame": "anchor_view_camera",
        "surface_completeness": (
            "Only surfaces visible in one or more supplied RGB-D views are reconstructed. "
            "Unseen regions remain unknown."
        ),
        "mesh_interpretation": (
            "The optional mesh is a voxel surface proxy for visualization only; "
            "it is not a watertight, collision-accurate object model."
        ),
        "anchor_view": str(anchor.path.resolve()),
        "input_views": [str(cloud.path.resolve()) for cloud in clouds],
        "view_registrations": registrations,
        "voxel_size_m": args.voxel_size,
        "max_depth_deviation_m": args.max_depth_deviation_m,
        "point_count": int(len(fused_points)),
        "bounds_anchor_m": {
            "min": fused_points.min(axis=0).tolist(),
            "max": fused_points.max(axis=0).tolist(),
        },
        "pointcloud_ply": str(ply_path.resolve()),
        "surface_proxy_mesh": str(args.mesh.resolve()) if args.mesh else None,
        "surface_proxy_faces": mesh_faces,
        "warnings": warnings,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Multi-view object surface reconstruction written")
    print(f"  views: {len(clouds)}, fused points: {len(fused_points)}")
    print(f"  fused npz: {args.output}")
    print(f"  fused ply: {ply_path}")
    if args.mesh:
        print(f"  surface-proxy mesh: {args.mesh} ({mesh_faces} faces)")
    print(f"  metadata: {metadata_path}")
    if warnings:
        print("  WARNING: " + "; ".join(warnings))


if __name__ == "__main__":
    main()
