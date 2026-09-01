"""Robot-independent canonical hand and grasp representation.

The retargeting optimizers consume 21 semantic landmarks, but a HUG prediction
contains substantially richer MANO information.  This module preserves that
information while providing a wrist-local, handedness-normalized 21x3 view
that can be shared by multiple robot morphology adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mediapipe import (
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    estimate_frame_from_hand_points,
)


SCHEMA_VERSION = 1
KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_INDICES = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)
BONE_EDGES = np.asarray(
    (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ),
    dtype=np.int64,
)
PINCH_PAIR_NAMES = (
    "thumb_index",
    "thumb_middle",
    "thumb_ring",
    "thumb_pinky",
)
PINCH_PAIRS = np.asarray(((4, 8), (4, 12), (4, 16), (4, 20)), dtype=np.int64)


def _as_array(value: Any, shape: tuple[int, ...], name: str, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def canonicalize_keypoints(
    keypoints_camera: np.ndarray, handedness: str = "right"
) -> tuple[np.ndarray, np.ndarray]:
    """Return wrist-local canonical landmarks and the row-vector basis.

    ``canonical = (camera - wrist) @ basis``.  The basis is derived solely
    from the human hand, never from a robot, so the result remains reusable
    across embodiments.  It deliberately matches AnyDexRetarget's existing
    hand-frame convention before any robot-specific rotation is applied.
    """
    handedness = handedness.lower()
    if handedness not in {"left", "right"}:
        raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
    points = _as_array(keypoints_camera, (21, 3), "keypoints_camera", np.float64)
    centered = points - points[0:1]
    wrist_frame = estimate_frame_from_hand_points(centered)
    operator = OPERATOR2MANO_RIGHT if handedness == "right" else OPERATOR2MANO_LEFT
    basis = wrist_frame @ operator
    canonical = centered @ basis
    return canonical.astype(np.float32), basis.astype(np.float32)


@dataclass(frozen=True)
class CanonicalGraspState:
    """A robot-independent static hand grasp state.

    The state retains the original HUG MANO data as evidence and exposes a
    canonical hand representation for downstream robot adapters.  It is not a
    contact optimizer: unknown contact fields are explicitly stored as NaN.
    """

    source: str
    handedness: str
    keypoints_camera: np.ndarray
    keypoints_canonical: np.ndarray
    wrist_position_camera: np.ndarray
    canonical_basis_row: np.ndarray
    bone_lengths: np.ndarray
    fingertip_positions_camera: np.ndarray
    fingertip_positions_canonical: np.ndarray
    pinch_distances: np.ndarray
    mano_pose: np.ndarray
    mano_pose_6d: np.ndarray
    mano_shape: np.ndarray
    mano_t_camera_wrist: np.ndarray
    mano_mesh_vertices_camera: np.ndarray
    mano_mesh_faces: np.ndarray
    condition_point_224: np.ndarray
    object_point_camera: np.ndarray
    object_point_canonical: np.ndarray
    fingertip_to_object_distance: np.ndarray

    @property
    def keypoints_wrist(self) -> np.ndarray:
        """Camera-oriented wrist-relative landmarks, before canonical rotation."""
        return self.keypoints_camera - self.wrist_position_camera[None]

    def keypoints_for_retargeting(self) -> np.ndarray:
        """Restore the camera-frame landmarks consumed by the Retargeter."""
        # The canonical state is the authority at this boundary. A future
        # refinement that updates keypoints_canonical therefore changes the
        # retargeting input while retaining the original wrist pose.
        basis = _as_array(
            self.canonical_basis_row, (3, 3), "canonical_basis_row", np.float64
        )
        canonical = _as_array(
            self.keypoints_canonical, (21, 3), "keypoints_canonical", np.float64
        )
        wrist = _as_array(
            self.wrist_position_camera, (3,), "wrist_position_camera", np.float64
        )
        restored = canonical @ basis.T + wrist[None]
        if not np.isfinite(restored).all():
            raise ValueError("Restored retargeting keypoints contain NaN or Inf")
        return restored

    def to_npz(self, path: str | Path) -> None:
        """Write a portable, named NumPy archive without pickled objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
            source=np.asarray(self.source),
            handedness=np.asarray(self.handedness),
            keypoint_names=np.asarray(KEYPOINT_NAMES),
            finger_names=np.asarray(FINGER_NAMES),
            bone_edges=BONE_EDGES,
            fingertip_indices=FINGERTIP_INDICES,
            pinch_pair_names=np.asarray(PINCH_PAIR_NAMES),
            pinch_pairs=PINCH_PAIRS,
            keypoints_camera=self.keypoints_camera,
            keypoints_wrist=self.keypoints_wrist,
            keypoints_canonical=self.keypoints_canonical,
            wrist_position_camera=self.wrist_position_camera,
            canonical_basis_row=self.canonical_basis_row,
            bone_lengths=self.bone_lengths,
            fingertip_positions_camera=self.fingertip_positions_camera,
            fingertip_positions_canonical=self.fingertip_positions_canonical,
            pinch_distances=self.pinch_distances,
            mano_pose=self.mano_pose,
            mano_pose_6d=self.mano_pose_6d,
            mano_shape=self.mano_shape,
            mano_t_camera_wrist=self.mano_t_camera_wrist,
            mano_mesh_vertices_camera=self.mano_mesh_vertices_camera,
            mano_mesh_faces=self.mano_mesh_faces,
            condition_point_224=self.condition_point_224,
            object_point_camera=self.object_point_camera,
            object_point_canonical=self.object_point_canonical,
            fingertip_to_object_distance=self.fingertip_to_object_distance,
        )


def canonical_grasp_from_hug(
    payload: Mapping[str, Any],
    *,
    handedness: str = "right",
    object_point_camera: np.ndarray | None = None,
    condition_point_224: np.ndarray | None = None,
) -> CanonicalGraspState:
    """Build a canonical state from an official HUG ``GraspData`` dictionary."""
    grasp = payload.get("grasp", payload)
    if not isinstance(grasp, Mapping):
        raise ValueError("HUG payload field 'grasp' must be a mapping")
    camera_points = _as_array(grasp.get("landmarks_3d"), (21, 3), "grasp.landmarks_3d")
    canonical, basis = canonicalize_keypoints(camera_points, handedness)
    wrist = camera_points[0].copy()
    bone_lengths = np.linalg.norm(
        canonical[BONE_EDGES[:, 1]] - canonical[BONE_EDGES[:, 0]], axis=1
    ).astype(np.float32)
    tips_camera = camera_points[FINGERTIP_INDICES].copy()
    tips_canonical = canonical[FINGERTIP_INDICES].copy()
    pinch_distances = np.linalg.norm(
        canonical[PINCH_PAIRS[:, 0]] - canonical[PINCH_PAIRS[:, 1]], axis=1
    ).astype(np.float32)

    object_camera = (
        np.full(3, np.nan, dtype=np.float32)
        if object_point_camera is None
        else _as_array(object_point_camera, (3,), "object_point_camera")
    )
    object_canonical = np.full(3, np.nan, dtype=np.float32)
    fingertip_distance = np.full(len(FINGERTIP_INDICES), np.nan, dtype=np.float32)
    if np.isfinite(object_camera).all():
        object_canonical = ((object_camera - wrist) @ basis).astype(np.float32)
        fingertip_distance = np.linalg.norm(tips_camera - object_camera[None], axis=1).astype(np.float32)

    condition = (
        np.full(2, np.nan, dtype=np.float32)
        if condition_point_224 is None
        else _as_array(condition_point_224, (2,), "condition_point_224")
    )
    return CanonicalGraspState(
        source="hug",
        handedness=handedness.lower(),
        keypoints_camera=camera_points,
        keypoints_canonical=canonical,
        wrist_position_camera=wrist,
        canonical_basis_row=basis,
        bone_lengths=bone_lengths,
        fingertip_positions_camera=tips_camera,
        fingertip_positions_canonical=tips_canonical,
        pinch_distances=pinch_distances,
        mano_pose=_as_array(grasp.get("pose"), (1, 15, 3), "grasp.pose"),
        mano_pose_6d=_as_array(grasp.get("pose_6d"), (1, 15, 6), "grasp.pose_6d"),
        mano_shape=_as_array(grasp.get("shape"), (1, 10), "grasp.shape"),
        mano_t_camera_wrist=_as_array(
            grasp.get("T_camera_wrist"), (4, 4), "grasp.T_camera_wrist"
        ),
        mano_mesh_vertices_camera=_as_array(
            grasp.get("mesh_vertices"), (778, 3), "grasp.mesh_vertices"
        ),
        mano_mesh_faces=np.asarray(grasp.get("mesh_faces"), dtype=np.int32).copy(),
        condition_point_224=condition,
        object_point_camera=object_camera,
        object_point_canonical=object_canonical,
        fingertip_to_object_distance=fingertip_distance,
    )


def load_canonical_grasp_state(path: str | Path) -> CanonicalGraspState:
    """Load a :class:`CanonicalGraspState` written by :meth:`to_npz`."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported canonical grasp schema version: {version}")
        return CanonicalGraspState(
            source=str(data["source"].item()),
            handedness=str(data["handedness"].item()),
            keypoints_camera=data["keypoints_camera"].copy(),
            keypoints_canonical=data["keypoints_canonical"].copy(),
            wrist_position_camera=data["wrist_position_camera"].copy(),
            canonical_basis_row=data["canonical_basis_row"].copy(),
            bone_lengths=data["bone_lengths"].copy(),
            fingertip_positions_camera=data["fingertip_positions_camera"].copy(),
            fingertip_positions_canonical=data["fingertip_positions_canonical"].copy(),
            pinch_distances=data["pinch_distances"].copy(),
            mano_pose=data["mano_pose"].copy(),
            mano_pose_6d=data["mano_pose_6d"].copy(),
            mano_shape=data["mano_shape"].copy(),
            mano_t_camera_wrist=data["mano_t_camera_wrist"].copy(),
            mano_mesh_vertices_camera=data["mano_mesh_vertices_camera"].copy(),
            mano_mesh_faces=data["mano_mesh_faces"].copy(),
            condition_point_224=data["condition_point_224"].copy(),
            object_point_camera=data["object_point_camera"].copy(),
            object_point_canonical=data["object_point_canonical"].copy(),
            fingertip_to_object_distance=data["fingertip_to_object_distance"].copy(),
        )


__all__ = [
    "BONE_EDGES",
    "CanonicalGraspState",
    "FINGERTIP_INDICES",
    "KEYPOINT_NAMES",
    "canonical_grasp_from_hug",
    "canonicalize_keypoints",
    "load_canonical_grasp_state",
]
