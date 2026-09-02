"""Backend-neutral conversion of metric object meshes into the grasp contract."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from anydexretarget.deployment import RECONSTRUCTION_SCHEMA_VERSION, rigid_transform


@dataclass(frozen=True)
class AlignmentResult:
    transform: np.ndarray
    rmse_m: float
    correspondences: int
    hypothesis: int | None


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Invalid reconstruction mesh: {path}")
    mesh = loaded.copy()
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("Mesh contains non-finite vertices")
    return mesh


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "points_camera" not in data:
            raise ValueError(f"Missing points_camera in {path}")
        points = np.asarray(data["points_camera"], dtype=np.float64)
        colors = np.asarray(data.get("colors_rgb", np.full(points.shape, 180)), dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(f"Invalid point cloud arrays: {points.shape}, {colors.shape}")
    if len(points) < 32 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Anchor cloud requires at least 32 finite positive-Z points")
    return points, colors


def load_transform(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = ("T_anchor_object", "object_to_anchor", "model_to_camera", "transform")
            key = next((candidate for candidate in keys if candidate in data), None)
            if key is None:
                raise ValueError(f"Transform NPZ must contain one of {keys}")
            value = np.asarray(data[key])
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("T_anchor_object", "object_to_anchor", "model_to_camera", "transform"):
            if key in payload:
                payload = payload[key]
                break
        value = np.asarray(payload)
    else:
        value = np.loadtxt(path)
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Transform must be a finite 4x4 matrix, got {transform.shape}")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError("Transform last row must be [0, 0, 0, 1]")
    rigid_transform(transform[:3, :3], transform[:3, 3])
    return transform


def _pca_axes(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _orientations() -> list[np.ndarray]:
    result = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base * np.asarray(signs)[None, :]
            if np.linalg.det(candidate) > 0:
                result.append(candidate)
    return result


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rigid_transform(rotation, target_center - rotation @ source_center)


def estimate_rigid_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 50,
    trim_fraction: float = 0.30,
    max_correspondence_m: float = 0.03,
) -> AlignmentResult:
    """Estimate anchor-from-object without changing the metric mesh scale."""
    target_tree = cKDTree(target)
    source_axes = _pca_axes(source)
    target_axes = _pca_axes(target)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    best: tuple[float, AlignmentResult] | None = None
    for hypothesis, signed in enumerate(_orientations()):
        rotation = target_axes @ signed @ source_axes.T
        transform = rigid_transform(rotation, target_center - rotation @ source_center)
        correspondences = 0
        for _ in range(iterations):
            moved = _apply(source, transform)
            distances, indices = target_tree.query(moved, k=1)
            valid = distances <= max_correspondence_m
            if valid.sum() < 32:
                break
            threshold = np.quantile(distances[valid], trim_fraction)
            valid &= distances <= threshold
            correspondences = int(valid.sum())
            if correspondences < 32:
                break
            transform = _kabsch(moved[valid], target[indices[valid]]) @ transform
        moved = _apply(source, transform)
        source_distances, _ = target_tree.query(moved, k=1)
        valid = source_distances <= max_correspondence_m
        if valid.sum() < 32:
            continue
        valid &= source_distances <= np.quantile(source_distances[valid], trim_fraction)
        rmse = float(np.sqrt(np.mean(source_distances[valid] ** 2)))
        visible_distances, _ = cKDTree(moved).query(target, k=1)
        score = rmse + float(np.quantile(visible_distances, 0.75))
        result = AlignmentResult(transform, rmse, int(valid.sum()), hypothesis)
        if best is None or score < best[0]:
            best = (score, result)
    if best is None:
        raise RuntimeError("Rigid registration failed; provide a measured T_anchor_object")
    return best[1]


def _surface_colors(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    try:
        colors = np.asarray(mesh.visual.to_color().face_colors, dtype=np.uint8)
        if colors.shape[0] == len(mesh.faces):
            return colors[face_indices, :3]
    except (AttributeError, ValueError):
        pass
    return np.full((len(face_indices), 3), 180, dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_reconstruction_mesh(
    *,
    mesh_path: Path,
    anchor_cloud_path: Path,
    output_dir: Path,
    backend: str,
    anchor_frame: str,
    transform: np.ndarray | None,
    auto_align: bool,
    source_unit_scale: float = 1.0,
    known_max_dimension_m: float | None = None,
    surface_samples: int = 20000,
    alignment_samples: int = 4000,
    merge_radius_m: float = 0.004,
    completed_confidence: float = 0.5,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    if (transform is None) == (not auto_align):
        raise ValueError("Choose exactly one of an explicit transform or auto_align")
    if source_unit_scale <= 0 or surface_samples < 1000 or alignment_samples < 1000:
        raise ValueError("Invalid unit scale or sample count")
    if known_max_dimension_m is not None and known_max_dimension_m <= 0:
        raise ValueError("known_max_dimension_m must be positive")
    if not 0 <= completed_confidence <= 1 or merge_radius_m <= 0:
        raise ValueError("Invalid confidence or merge radius")

    outputs = {
        "mesh": output_dir / "object_mesh_anchor.ply",
        "mesh_cloud": output_dir / "mesh_surface_anchor.npz",
        "surface": output_dir / "object_surface_anchor.npz",
        "alignment": output_dir / "alignment_report.json",
        "metadata": output_dir / "reconstruction_metadata.json",
    }
    if not overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    mesh = load_mesh(mesh_path)
    mesh.apply_scale(source_unit_scale)
    extent_before_known_scale = np.asarray(mesh.extents, dtype=np.float64)
    known_scale = 1.0
    if known_max_dimension_m is not None:
        known_scale = known_max_dimension_m / float(extent_before_known_scale.max())
        mesh.apply_scale(known_scale)
    visible_points, visible_colors = load_cloud(anchor_cloud_path)
    np.random.seed(seed)
    registration_source, _ = trimesh.sample.sample_surface(mesh, alignment_samples)
    if auto_align:
        alignment = estimate_rigid_alignment(registration_source, visible_points)
    else:
        assert transform is not None
        moved = _apply(registration_source, transform)
        distances, _ = cKDTree(moved).query(visible_points, k=1)
        alignment = AlignmentResult(
            transform=transform,
            rmse_m=float(np.sqrt(np.mean(distances ** 2))),
            correspondences=int(len(distances)),
            hypothesis=None,
        )
    mesh.apply_transform(alignment.transform)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if np.any(vertices[:, 2] <= 0):
        raise ValueError("Aligned mesh crosses/behind anchor optical camera (Z <= 0)")

    mesh_points, face_indices = trimesh.sample.sample_surface(mesh, surface_samples)
    mesh_colors = _surface_colors(mesh, face_indices)
    visible_tree = cKDTree(visible_points)
    distances, nearest = visible_tree.query(mesh_points, k=1)
    completion = distances > merge_radius_m
    completion_points = mesh_points[completion]
    completion_colors = mesh_colors[completion]
    untextured = np.all(completion_colors == 180, axis=1)
    completion_colors[untextured] = visible_colors[nearest[completion][untextured]]
    points = np.concatenate((visible_points, completion_points), axis=0).astype(np.float32)
    colors = np.concatenate((visible_colors, completion_colors), axis=0).astype(np.uint8)
    observed = np.concatenate((np.ones(len(visible_points)), np.zeros(len(completion_points)))).astype(np.uint8)
    confidence = np.where(observed.astype(bool), 1.0, completed_confidence).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(outputs["mesh"])
    np.savez_compressed(
        outputs["mesh_cloud"],
        points_camera=mesh_points.astype(np.float32),
        colors_rgb=mesh_colors.astype(np.uint8),
        model_to_camera=alignment.transform,
        source_type=np.asarray(f"{backend}_mesh_aligned_to_anchor"),
    )
    np.savez_compressed(
        outputs["surface"],
        schema_version=np.asarray(RECONSTRUCTION_SCHEMA_VERSION, dtype=np.int64),
        points_camera=points,
        colors_rgb=colors,
        confidence=confidence,
        is_observed=observed,
        coordinate_frame=np.asarray("anchor_optical_camera"),
        anchor_frame=np.asarray(anchor_frame),
        backend=np.asarray(backend),
        source_mesh=np.asarray(str(mesh_path.resolve())),
        source_pointcloud=np.asarray(str(anchor_cloud_path.resolve())),
    )
    alignment_report = {
        "mode": "auto_rigid" if auto_align else "explicit_transform",
        "transform_convention": "T_anchor_object maps metric object-frame points into anchor optical frame",
        "T_anchor_object": alignment.transform.tolist(),
        "source_unit_scale_to_meter": source_unit_scale,
        "known_dimension_scale": known_scale,
        "source_extent_after_unit_conversion_m": extent_before_known_scale.tolist(),
        "aligned_extent_m": np.asarray(mesh.extents).tolist(),
        "visible_surface_rmse_m": alignment.rmse_m,
        "correspondences": alignment.correspondences,
        "pca_hypothesis": alignment.hypothesis,
        "requires_visual_overlay_check": bool(auto_align),
    }
    outputs["alignment"].write_text(json.dumps(alignment_report, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "backend": backend,
        "anchor_frame": anchor_frame,
        "units": "meter",
        "coordinate_convention": "OpenCV optical: +X right, +Y down, +Z forward",
        "mesh": outputs["mesh"].name,
        "surface_pointcloud": outputs["surface"].name,
        "alignment_report": outputs["alignment"].name,
        "source_mesh": str(mesh_path.resolve()),
        "source_anchor_pointcloud": str(anchor_cloud_path.resolve()),
        "source_sha256": {"mesh": _sha256(mesh_path), "anchor_pointcloud": _sha256(anchor_cloud_path)},
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "visible_measured_points": int(len(visible_points)),
        "completed_mesh_points": int(len(completion_points)),
        "surface_points": int(len(points)),
        "alignment": alignment_report,
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "metadata": metadata}


__all__ = [
    "AlignmentResult",
    "adapt_reconstruction_mesh",
    "estimate_rigid_alignment",
    "load_cloud",
    "load_mesh",
    "load_transform",
]
