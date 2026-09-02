"""Robot-independent hand frame shared by live and static input sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand_representation import canonicalize_keypoints


@dataclass(frozen=True)
class CanonicalHandFrame:
    """One timestamped hand pose before robot-specific retargeting."""

    source: str
    handedness: str
    timestamp_s: float
    keypoints_21: np.ndarray
    keypoints_canonical: np.ndarray
    wrist_position: np.ndarray
    canonical_basis_row: np.ndarray
    valid_mask_21: np.ndarray
    source_keypoints: np.ndarray
    source_joint_angles: np.ndarray

    def keypoints_for_retargeting(self) -> np.ndarray:
        """Return the standard 21x3 input expected by all retargeting backends."""
        restored = (
            np.asarray(self.keypoints_canonical, dtype=np.float64)
            @ np.asarray(self.canonical_basis_row, dtype=np.float64).T
            + np.asarray(self.wrist_position, dtype=np.float64)[None]
        )
        if restored.shape != (21, 3) or not np.isfinite(restored).all():
            raise ValueError("Restored hand keypoints must be finite with shape (21, 3)")
        return restored.astype(np.float32)


def canonical_hand_frame(
    keypoints_21: np.ndarray,
    *,
    source: str,
    handedness: str,
    timestamp_s: float = 0.0,
    valid_mask_21: np.ndarray | None = None,
    source_keypoints: np.ndarray | None = None,
    source_joint_angles: np.ndarray | None = None,
) -> CanonicalHandFrame:
    """Validate and canonicalize one standard 21-point hand frame."""
    points = np.asarray(keypoints_21, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("keypoints_21 must be finite with shape (21, 3)")
    mask = (
        np.ones(21, dtype=bool)
        if valid_mask_21 is None
        else np.asarray(valid_mask_21, dtype=bool)
    )
    if mask.shape != (21,) or not mask.all():
        raise ValueError("All 21 required hand keypoints must be valid")
    canonical, basis = canonicalize_keypoints(points, handedness)
    source_points = (
        points if source_keypoints is None else np.asarray(source_keypoints, dtype=np.float64)
    )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_keypoints must be an Nx3 array")
    joint_angles = (
        np.empty(0, dtype=np.float32)
        if source_joint_angles is None
        else np.asarray(source_joint_angles, dtype=np.float32)
    )
    if joint_angles.ndim != 1 or not np.isfinite(joint_angles).all():
        raise ValueError("source_joint_angles must be a finite one-dimensional array")
    return CanonicalHandFrame(
        source=str(source),
        handedness=handedness.lower(),
        timestamp_s=float(timestamp_s),
        keypoints_21=points.astype(np.float32),
        keypoints_canonical=canonical,
        wrist_position=points[0].astype(np.float32),
        canonical_basis_row=basis,
        valid_mask_21=mask,
        source_keypoints=source_points.astype(np.float32),
        source_joint_angles=joint_angles.copy(),
    )


__all__ = ["CanonicalHandFrame", "canonical_hand_frame"]
