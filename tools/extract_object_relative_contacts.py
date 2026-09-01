#!/usr/bin/env python3
"""Extract object-relative fingertip anchors from one HUG canonical grasp.

The output is an evidence record, not a force-closure claim.  A fingertip is
marked ``near_surface`` only when its HUG prediction lies within the requested
gap of the aligned object mesh; no physical contact sensor is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from anydexretarget.hand_representation import FINGER_NAMES, load_canonical_grasp_state


SCHEMA_VERSION = 1


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [item for item in raw.geometry.values() if isinstance(item, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    return raw


def _object_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Use a reproducible provisional object frame without inventing axes."""
    origin = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    object_to_camera = np.eye(4, dtype=np.float64)
    object_to_camera[:3, 3] = origin
    camera_to_object = np.linalg.inv(object_to_camera)
    return object_to_camera, camera_to_object


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _outward_normals(mesh: trimesh.Trimesh, anchors: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64).copy()
    center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    flip = np.sum(normals * (anchors - center[None]), axis=1) < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-grasp", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True,
                        help="Photo-aligned object mesh in the same camera frame as HUG.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    args = parser.parse_args()
    if args.near_surface_gap_mm <= 0:
        raise ValueError("--near-surface-gap-mm must be positive")

    state = load_canonical_grasp_state(args.canonical_grasp)
    mesh = _load_mesh(args.object_mesh)
    tips_camera = np.asarray(state.fingertip_positions_camera, dtype=np.float64)
    if tips_camera.shape != (5, 3) or not np.isfinite(tips_camera).all():
        raise ValueError("Canonical grasp has invalid fingertip positions")
    anchors_camera, distances, face_indices = trimesh.proximity.closest_point_naive(mesh, tips_camera)
    anchors_camera = np.asarray(anchors_camera, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    face_indices = np.asarray(face_indices, dtype=np.int32)
    normals_camera = _outward_normals(mesh, anchors_camera, face_indices)
    near_surface = distances <= args.near_surface_gap_mm / 1000.0

    object_to_camera, camera_to_object = _object_frame(mesh)
    tips_object = _transform_points(tips_camera, camera_to_object)
    anchors_object = _transform_points(anchors_camera, camera_to_object)
    normals_object = normals_camera @ camera_to_object[:3, :3].T
    wrist_object = _transform_points(state.wrist_position_camera[None], camera_to_object)[0]
    mano_t_object_wrist = camera_to_object @ np.asarray(state.mano_t_camera_wrist, dtype=np.float64)
    offsets = tips_camera - anchors_camera

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        source_canonical_grasp=np.asarray(str(args.canonical_grasp.resolve())),
        source_object_mesh=np.asarray(str(args.object_mesh.resolve())),
        object_frame_definition=np.asarray("origin=mesh vertex centroid; axes=anchor camera axes"),
        object_to_camera=object_to_camera,
        camera_to_object=camera_to_object,
        mano_t_object_wrist=mano_t_object_wrist,
        wrist_position_object=wrist_object.astype(np.float32),
        finger_names=np.asarray(FINGER_NAMES),
        fingertip_positions_camera=tips_camera.astype(np.float32),
        fingertip_positions_object=tips_object.astype(np.float32),
        surface_anchor_camera=anchors_camera.astype(np.float32),
        surface_anchor_object=anchors_object.astype(np.float32),
        surface_normal_camera=normals_camera.astype(np.float32),
        surface_normal_object=normals_object.astype(np.float32),
        nearest_face_index=face_indices,
        tip_surface_distance_m=distances.astype(np.float32),
        fingertip_to_anchor_offset_camera=offsets.astype(np.float32),
        near_surface=near_surface.astype(np.uint8),
        near_surface_gap_m=np.asarray(args.near_surface_gap_mm / 1000.0, dtype=np.float32),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_canonical_grasp": str(args.canonical_grasp.resolve()),
        "source_object_mesh": str(args.object_mesh.resolve()),
        "coordinate_frame": "anchor_rgbd_camera",
        "object_frame_definition": "origin=mesh vertex centroid; axes=anchor camera axes",
        "mesh_watertight": bool(mesh.is_watertight),
        "near_surface_gap_mm": args.near_surface_gap_mm,
        "contacts": [
            {
                "finger": name,
                "tip_surface_distance_mm": float(distance * 1000.0),
                "near_surface": bool(is_near),
                "surface_face": int(face),
            }
            for name, distance, is_near, face in zip(FINGER_NAMES, distances, near_surface, face_indices)
        ],
        "near_surface_count": int(near_surface.sum()),
        "note": "Near-surface means geometric proximity to a reconstructed mesh. It is not verified contact, force closure, collision-free approach, or physical stability.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Object-relative contact plan written")
    print(f"  near-surface fingertips: {int(near_surface.sum())}/5")
    print("  distances [mm]: " + ", ".join(f"{name}={distance * 1000:.1f}" for name, distance in zip(FINGER_NAMES, distances)))
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
