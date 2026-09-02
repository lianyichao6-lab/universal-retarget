#!/usr/bin/env python3
"""Package any aligned reconstruction into the common downstream contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from anydexretarget.deployment import RECONSTRUCTION_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No triangle geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Invalid reconstruction mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if not np.isfinite(vertices).all() or np.any(vertices[:, 2] <= 0):
        raise ValueError("Mesh must be finite and in front of the anchor optical camera")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--surface-pointcloud", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--anchor-frame", default="waist_camera_color_optical_frame")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_output = args.output_dir / "object_mesh_anchor.ply"
    cloud_output = args.output_dir / "object_surface_anchor.npz"
    metadata_output = args.output_dir / "reconstruction_metadata.json"
    existing = [path for path in (mesh_output, cloud_output, metadata_output) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite reconstruction result: "
            + ", ".join(str(path) for path in existing)
        )
    mesh = _mesh(args.mesh)
    with np.load(args.surface_pointcloud, allow_pickle=False) as data:
        if "points_camera" not in data or "colors_rgb" not in data:
            raise ValueError("Surface point cloud requires points_camera and colors_rgb")
        points = np.asarray(data["points_camera"], dtype=np.float32)
        colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
        observed = (
            np.asarray(data["is_observed"], dtype=np.uint8)
            if "is_observed" in data
            else np.zeros(len(points), dtype=np.uint8)
        )
        confidence = (
            np.asarray(data["confidence"], dtype=np.float32)
            if "confidence" in data
            else np.where(observed.astype(bool), 1.0, 0.5).astype(np.float32)
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or observed.shape != (len(points),)
        or confidence.shape != (len(points),)
        or len(points) == 0
        or not np.isfinite(points).all()
        or not np.isfinite(confidence).all()
        or np.any(points[:, 2] <= 0)
        or np.any((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("Invalid anchor-frame reconstruction point cloud")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_output)
    np.savez_compressed(
        cloud_output,
        schema_version=np.asarray(RECONSTRUCTION_SCHEMA_VERSION, dtype=np.int64),
        points_camera=points,
        colors_rgb=colors,
        confidence=confidence,
        is_observed=observed,
        coordinate_frame=np.asarray("anchor_optical_camera"),
        anchor_frame=np.asarray(args.anchor_frame),
        backend=np.asarray(args.backend),
        source_mesh=np.asarray(str(args.mesh.resolve())),
        source_pointcloud=np.asarray(str(args.surface_pointcloud.resolve())),
    )
    source_metadata = None
    if args.source_metadata is not None:
        source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    report = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "backend": args.backend,
        "anchor_frame": args.anchor_frame,
        "units": "meter",
        "coordinate_convention": "OpenCV optical: +X right, +Y down, +Z forward",
        "mesh": mesh_output.name,
        "surface_pointcloud": cloud_output.name,
        "source_mesh": str(args.mesh.resolve()),
        "source_pointcloud": str(args.surface_pointcloud.resolve()),
        "source_sha256": {
            "mesh": _sha256(args.mesh),
            "pointcloud": _sha256(args.surface_pointcloud),
        },
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "surface_points": int(len(points)),
        "observed_points": int(observed.astype(bool).sum()),
        "completed_points": int((~observed.astype(bool)).sum()),
        "bounds_anchor_m": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "source_metadata": source_metadata,
    }
    metadata_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Portable reconstruction result written")
    print(f"  backend: {args.backend}")
    print(f"  mesh: {mesh_output}")
    print(f"  HUG surface: {cloud_output}")
    print(f"  metadata: {metadata_output}")


if __name__ == "__main__":
    main()
