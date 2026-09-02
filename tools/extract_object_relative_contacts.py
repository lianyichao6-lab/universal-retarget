#!/usr/bin/env python3
"""Extract distal-pad object-relative contacts from one HUG canonical grasp.

Each finger is sampled at three positions along its distal phalanx. The closest
sample supplies a provisional surface anchor and normal. These are geometric
observations, not physical contact or force-closure claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from anydexretarget.hand_representation import FINGER_NAMES, load_canonical_grasp_state


SCHEMA_VERSION = 2
DISTAL_KEYPOINT_INDICES = np.asarray((3, 7, 11, 15, 19), dtype=np.int64)
PAD_SAMPLE_ALPHAS = np.asarray((0.35, 0.65, 1.0), dtype=np.float64)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [item for item in raw.geometry.values() if isinstance(item, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    return raw


def _object_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    object_to_camera = np.eye(4, dtype=np.float64)
    object_to_camera[:3, 3] = origin
    return object_to_camera, np.linalg.inv(object_to_camera)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _outward_normals(
    mesh: trimesh.Trimesh, anchors: np.ndarray, face_indices: np.ndarray
) -> np.ndarray:
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64).copy()
    center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    flip = np.sum(normals * (anchors - center[None]), axis=1) < 0.0
    normals[flip] *= -1.0
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-grasp", type=Path, required=True)
    parser.add_argument(
        "--object-mesh",
        type=Path,
        required=True,
        help="Photo-aligned object mesh in the same camera frame as HUG.",
    )
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
    distal_camera = np.asarray(
        state.keypoints_camera[DISTAL_KEYPOINT_INDICES], dtype=np.float64
    )
    pad_samples_camera = (
        distal_camera[:, None, :]
        + PAD_SAMPLE_ALPHAS[None, :, None]
        * (tips_camera - distal_camera)[:, None, :]
    )
    sample_anchors, sample_distances, sample_faces = (
        trimesh.proximity.closest_point_naive(mesh, pad_samples_camera.reshape(-1, 3))
    )
    sample_anchors = np.asarray(sample_anchors, dtype=np.float64).reshape(5, -1, 3)
    sample_distances = np.asarray(sample_distances, dtype=np.float64).reshape(5, -1)
    sample_faces = np.asarray(sample_faces, dtype=np.int32).reshape(5, -1)
    selected_sample = np.argmin(sample_distances, axis=1)
    finger_indices = np.arange(len(FINGER_NAMES))
    contact_points_camera = pad_samples_camera[finger_indices, selected_sample]
    contact_alphas = PAD_SAMPLE_ALPHAS[selected_sample]
    anchors_camera = sample_anchors[finger_indices, selected_sample]
    distances = sample_distances[finger_indices, selected_sample]
    face_indices = sample_faces[finger_indices, selected_sample]
    normals_camera = _outward_normals(mesh, anchors_camera, face_indices)
    near_surface = distances <= args.near_surface_gap_mm / 1000.0

    tip_anchors_camera, tip_distances, tip_face_indices = (
        trimesh.proximity.closest_point_naive(mesh, tips_camera)
    )
    tip_anchors_camera = np.asarray(tip_anchors_camera, dtype=np.float64)
    tip_distances = np.asarray(tip_distances, dtype=np.float64)
    tip_face_indices = np.asarray(tip_face_indices, dtype=np.int32)

    object_to_camera, camera_to_object = _object_frame(mesh)
    tips_object = _transform_points(tips_camera, camera_to_object)
    contact_points_object = _transform_points(contact_points_camera, camera_to_object)
    anchors_object = _transform_points(anchors_camera, camera_to_object)
    normals_object = normals_camera @ camera_to_object[:3, :3].T
    wrist_object = _transform_points(
        state.wrist_position_camera[None], camera_to_object
    )[0]
    mano_t_object_wrist = (
        camera_to_object @ np.asarray(state.mano_t_camera_wrist, dtype=np.float64)
    )
    contact_offsets = contact_points_camera - anchors_camera
    fingertip_offsets = tips_camera - tip_anchors_camera

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        source_canonical_grasp=np.asarray(str(args.canonical_grasp.resolve())),
        source_object_mesh=np.asarray(str(args.object_mesh.resolve())),
        object_frame_definition=np.asarray(
            "origin=mesh vertex centroid; axes=anchor camera axes"
        ),
        object_to_camera=object_to_camera,
        camera_to_object=camera_to_object,
        mano_t_object_wrist=mano_t_object_wrist,
        wrist_position_object=wrist_object.astype(np.float32),
        finger_names=np.asarray(FINGER_NAMES),
        fingertip_positions_camera=tips_camera.astype(np.float32),
        fingertip_positions_object=tips_object.astype(np.float32),
        distal_keypoint_indices=DISTAL_KEYPOINT_INDICES,
        distal_positions_camera=distal_camera.astype(np.float32),
        pad_sample_alphas=PAD_SAMPLE_ALPHAS.astype(np.float32),
        pad_sample_positions_camera=pad_samples_camera.astype(np.float32),
        pad_sample_surface_anchor_camera=sample_anchors.astype(np.float32),
        pad_sample_surface_distance_m=sample_distances.astype(np.float32),
        pad_sample_nearest_face_index=sample_faces,
        selected_pad_sample_index=selected_sample.astype(np.int64),
        contact_point_alpha=contact_alphas.astype(np.float32),
        contact_point_kind=np.asarray(
            [
                "tip" if np.isclose(alpha, 1.0) else "distal_pad"
                for alpha in contact_alphas
            ]
        ),
        contact_point_positions_camera=contact_points_camera.astype(np.float32),
        contact_point_positions_object=contact_points_object.astype(np.float32),
        surface_anchor_camera=anchors_camera.astype(np.float32),
        surface_anchor_object=anchors_object.astype(np.float32),
        surface_normal_camera=normals_camera.astype(np.float32),
        surface_normal_object=normals_object.astype(np.float32),
        nearest_face_index=face_indices,
        contact_surface_distance_m=distances.astype(np.float32),
        contact_to_anchor_offset_camera=contact_offsets.astype(np.float32),
        tip_surface_anchor_camera=tip_anchors_camera.astype(np.float32),
        tip_surface_distance_m=tip_distances.astype(np.float32),
        tip_nearest_face_index=tip_face_indices,
        # Compatibility aliases for v1 readers.
        fingertip_to_anchor_offset_camera=fingertip_offsets.astype(np.float32),
        near_surface=near_surface.astype(np.uint8),
        near_surface_gap_m=np.asarray(
            args.near_surface_gap_mm / 1000.0, dtype=np.float32
        ),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_canonical_grasp": str(args.canonical_grasp.resolve()),
        "source_object_mesh": str(args.object_mesh.resolve()),
        "coordinate_frame": "anchor_rgbd_camera",
        "mesh_watertight": bool(mesh.is_watertight),
        "near_surface_gap_mm": args.near_surface_gap_mm,
        "contacts": [
            {
                "finger": name,
                "contact_point_kind": (
                    "tip" if np.isclose(alpha, 1.0) else "distal_pad"
                ),
                "contact_point_alpha": float(alpha),
                "contact_surface_distance_mm": float(distance * 1000.0),
                "tip_surface_distance_mm": float(tip_distance * 1000.0),
                "near_surface": bool(is_near),
                "surface_face": int(face),
            }
            for name, alpha, distance, tip_distance, is_near, face in zip(
                FINGER_NAMES,
                contact_alphas,
                distances,
                tip_distances,
                near_surface,
                face_indices,
            )
        ],
        "near_surface_count": int(near_surface.sum()),
        "note": (
            "Each finger selects the closest of three distal-pad samples. "
            "Near-surface remains geometric evidence, not verified contact, "
            "force closure, collision-free approach, or physical stability."
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Object-relative distal-pad contact plan written")
    print(f"  near-surface fingers: {int(near_surface.sum())}/5")
    print(
        "  selected contacts [mm]: "
        + ", ".join(
            "{}:{}={:.1f}".format(name, "tip" if np.isclose(alpha, 1.0) else "pad", distance * 1000.0)
            for name, alpha, distance in zip(
                FINGER_NAMES, contact_alphas, distances
            )
        )
    )
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
