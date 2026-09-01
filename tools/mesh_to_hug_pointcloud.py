#!/usr/bin/env python3
"""Convert an aligned CAD mesh into the external point-cloud format used by HUG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _parse_color(values: list[int]) -> np.ndarray:
    color = np.asarray(values, dtype=np.int64)
    if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
        raise argparse.ArgumentTypeError("--color values must be three integers in [0, 255]")
    return color.astype(np.uint8)


def _load_transform(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        matrix = np.load(path)
    else:
        matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite 4x4 model_to_camera matrix, got {matrix.shape}: {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("model_to_camera last row must be [0, 0, 0, 1]")
    return matrix


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Invalid triangle mesh: {path}")
    return loaded


def _sample_colors(mesh: trimesh.Trimesh, face_index: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    visual = mesh.visual
    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors) == len(mesh.faces):
        return np.asarray(face_colors[face_index, :3], dtype=np.uint8)
    return np.broadcast_to(fallback, (len(face_index), 3)).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Source OBJ/PLY/STL mesh in model coordinates.")
    parser.add_argument(
        "--model-to-camera", type=Path, required=True,
        help="4x4 text or .npy transform from meter-scaled model coordinates to anchor camera coordinates.",
    )
    parser.add_argument("--model-unit", choices=sorted(UNIT_TO_METERS), default="m")
    parser.add_argument("--samples", type=int, default=20000, help="Number of uniformly sampled surface points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color", type=int, nargs=3, default=(180, 180, 180), metavar=("R", "G", "B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 32:
        parser.error("--samples must be at least 32")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite {args.output}; pass --overwrite")

    np.random.seed(args.seed)
    mesh = _as_mesh(args.mesh)
    unit_scale = UNIT_TO_METERS[args.model_unit]
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * unit_scale
    transform = _load_transform(args.model_to_camera)
    points_model, face_index = trimesh.sample.sample_surface(mesh, args.samples)
    points_h = np.c_[points_model, np.ones(len(points_model), dtype=np.float64)]
    points_camera = (points_h @ transform.T)[:, :3]
    if not np.isfinite(points_camera).all() or np.any(points_camera[:, 2] <= 0):
        raise ValueError("Transformed mesh has non-finite or non-positive camera Z points; check model_to_camera")
    colors = _sample_colors(mesh, face_index, _parse_color(args.color))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_camera=points_camera.astype(np.float32),
        colors_rgb=colors.astype(np.uint8),
        source_type=np.asarray("aligned_cad_mesh"),
        source_mesh=np.asarray(str(args.mesh.resolve())),
        model_unit=np.asarray(args.model_unit),
        model_to_camera=transform.astype(np.float64),
    )
    metadata = {
        "source_type": "aligned_cad_mesh",
        "source_mesh": str(args.mesh.resolve()),
        "coordinate_frame": "anchor_rgbd_camera",
        "model_unit": args.model_unit,
        "unit_scale_to_meters": unit_scale,
        "model_to_camera": transform.tolist(),
        "surface_samples": int(args.samples),
        "camera_bounds_m": {
            "min": np.min(points_camera, axis=0).tolist(),
            "max": np.max(points_camera, axis=0).tolist(),
        },
        "note": "HUG consumes points_camera/colors_rgb. The mesh must be aligned to the anchor RGB-D camera before use.",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"HUG CAD point cloud written: {args.output}")
    print(f"  points: {len(points_camera)}, source faces: {len(mesh.faces)}")
    print(f"  camera Z range: {points_camera[:, 2].min():.4f} .. {points_camera[:, 2].max():.4f} m")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
