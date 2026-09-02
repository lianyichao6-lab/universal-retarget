"""Convert Luban MANUS semantic keypoints to the standard 21-point layout."""

from __future__ import annotations

import numpy as np

from .hand_frame import CanonicalHandFrame, canonical_hand_frame


MANUS_KEYPOINT_COUNT = 25

# MANUS uses five slots per finger: metacarpal, proximal, intermediate,
# distal, tip. The thumb has no intermediate node.
MANUS_TO_21 = np.asarray(
    (
        0, 1, 3, 4,
        6, 7, 8, 9,
        11, 12, 13, 14,
        16, 17, 18, 19,
        21, 22, 23, 24,
    ),
    dtype=np.int64,
)


def manus_keypoints_to_21(
    wrist: np.ndarray,
    keypoints_25: np.ndarray,
    keypoint_mask: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standard wrist+five-finger 21x3 points and their valid mask."""
    wrist = np.asarray(wrist, dtype=np.float64)
    points = np.asarray(keypoints_25, dtype=np.float64)
    if wrist.shape != (3,) or not np.isfinite(wrist).all():
        raise ValueError("MANUS wrist must be a finite 3-vector")
    if points.shape != (MANUS_KEYPOINT_COUNT, 3):
        raise ValueError("MANUS keypoints must have shape (25, 3)")
    selected = points[MANUS_TO_21]
    selected_valid = np.asarray(
        [bool((int(keypoint_mask) >> int(index)) & 1) for index in MANUS_TO_21],
        dtype=bool,
    )
    valid = np.concatenate((np.ones(1, dtype=bool), selected_valid))
    output = np.vstack((wrist[None], selected))
    if not np.isfinite(output[valid]).all():
        valid[1:] &= np.isfinite(selected).all(axis=1)
    return output.astype(np.float32), valid


def canonical_hand_frame_from_manus(
    wrist: np.ndarray,
    keypoints_25: np.ndarray,
    keypoint_mask: int,
    *,
    handedness: str,
    timestamp_s: float = 0.0,
    ergonomics: np.ndarray | None = None,
) -> CanonicalHandFrame:
    """Build a robot-independent frame from one valid MANUS hand sample."""
    points_21, valid = manus_keypoints_to_21(wrist, keypoints_25, keypoint_mask)
    if not valid.all():
        missing = np.flatnonzero(~valid).tolist()
        raise ValueError(f"MANUS frame is missing required 21-point landmarks: {missing}")
    return canonical_hand_frame(
        points_21,
        source="manus",
        handedness=handedness,
        timestamp_s=timestamp_s,
        valid_mask_21=valid,
        source_keypoints=keypoints_25,
        source_joint_angles=ergonomics,
    )


__all__ = [
    "MANUS_KEYPOINT_COUNT",
    "MANUS_TO_21",
    "canonical_hand_frame_from_manus",
    "manus_keypoints_to_21",
]
