"""Continuous, morphology-conditioned target chains for LinkerHand L25.

Vector retargeting optimizes independent palm-to-landmark vectors.  This
module deliberately does not alter that optimizer.  It derives a separate,
continuous target skeleton from the complete MANO landmark chain: every human
bone contributes a direction and every L25 segment contributes its measured
zero-pose FK length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FINGER_CHAINS = np.asarray(
    (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    ),
    dtype=np.int64,
)


@dataclass(frozen=True)
class L25TargetChain:
    """A 21-point L25-length target skeleton derived from MANO landmarks."""

    target_points: np.ndarray
    reference_points: np.ndarray
    segment_lengths: np.ndarray
    source_directions: np.ndarray
    source_keypoints: np.ndarray


def _point_position(robot, link_name: str, offset: np.ndarray | None = None) -> np.ndarray:
    link_id = robot.get_link_index(link_name)
    pose = robot.get_link_pose(int(link_id))
    point = pose[:3, 3].copy()
    if offset is not None:
        point += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
    return point


def _reference_qpos(optimizer) -> np.ndarray:
    qpos = (
        optimizer.neutral_qpos.copy()
        if optimizer.neutral_qpos is not None
        else np.zeros(optimizer.robot.model.nq, dtype=np.float64)
    )
    lower, upper = optimizer.robot.joint_limits[:, 0], optimizer.robot.joint_limits[:, 1]
    return np.clip(qpos, lower, upper)


def l25_chain_points(optimizer, qpos: np.ndarray) -> np.ndarray:
    """Return L25 FK points in the 21-point MANO/MediaPipe semantic layout."""
    if optimizer.num_fingers != 5:
        raise ValueError("L25 target chains require five configured fingers")
    qpos = np.asarray(qpos, dtype=np.float64)
    optimizer.robot.compute_forward_kinematics(qpos)
    points = np.full((21, 3), np.nan, dtype=np.float64)
    origin = _point_position(optimizer.robot, optimizer.origin_link_name)
    points[0] = origin
    for finger, indices in enumerate(FINGER_CHAINS):
        chain = (
            _point_position(optimizer.robot, optimizer.link1_names[finger]),
            _point_position(
                optimizer.robot, optimizer.link3_names[finger], optimizer.link3_offsets[finger]
            ),
            _point_position(
                optimizer.robot, optimizer.link4_names[finger], optimizer.link4_offsets[finger]
            ),
            _point_position(
                optimizer.robot, optimizer.task_link_names[finger], optimizer.task_offsets[finger]
            ),
        )
        points[indices[1:]] = np.asarray(chain)
    if not np.isfinite(points).all():
        raise ValueError("L25 FK chain contains NaN or Inf")
    return points


def build_l25_target_chain(optimizer, transformed_keypoints: np.ndarray) -> L25TargetChain:
    """Build a continuous L25-length target from the complete MANO chain.

    ``transformed_keypoints`` must be the exact robot-frame hand landmarks
    produced by ``Retargeter.retarget_verbose(...)[1]['mediapipe_kp']``. This
    preserves the configured coordinate conversion while avoiding any vector
    scale approximation.
    """
    source = np.asarray(transformed_keypoints, dtype=np.float64)
    if source.shape != (21, 3) or not np.isfinite(source).all():
        raise ValueError("transformed_keypoints must be finite with shape (21, 3)")

    reference = l25_chain_points(optimizer, _reference_qpos(optimizer))
    target = np.empty_like(reference)
    target[0] = reference[0]
    segment_lengths = np.empty((5, 4), dtype=np.float64)
    source_directions = np.empty((5, 4, 3), dtype=np.float64)

    for finger, indices in enumerate(FINGER_CHAINS):
        for segment in range(4):
            source_vector = source[indices[segment + 1]] - source[indices[segment]]
            source_length = float(np.linalg.norm(source_vector))
            if source_length <= 1e-8:
                raise ValueError(
                    f"MANO segment is degenerate for finger {finger}, segment {segment}"
                )
            direction = source_vector / source_length
            robot_vector = (
                reference[indices[segment + 1]] - reference[indices[segment]]
            )
            robot_length = float(np.linalg.norm(robot_vector))
            if robot_length <= 1e-8:
                raise ValueError(
                    f"L25 reference segment is degenerate for finger {finger}, segment {segment}"
                )
            source_directions[finger, segment] = direction
            segment_lengths[finger, segment] = robot_length
            target[indices[segment + 1]] = target[indices[segment]] + robot_length * direction

    return L25TargetChain(
        target_points=target,
        reference_points=reference,
        segment_lengths=segment_lengths,
        source_directions=source_directions,
        source_keypoints=source.copy(),
    )


def chain_error_metrics(target: L25TargetChain, robot_points: np.ndarray) -> dict[str, float]:
    """Return position, direction, and thumb-index error for a solved L25 pose."""
    robot = np.asarray(robot_points, dtype=np.float64)
    if robot.shape != (21, 3) or not np.isfinite(robot).all():
        raise ValueError("robot_points must be finite with shape (21, 3)")
    position_errors = np.linalg.norm(robot - target.target_points, axis=1)
    direction_angles = []
    for indices in FINGER_CHAINS:
        for segment in range(4):
            desired = target.target_points[indices[segment + 1]] - target.target_points[indices[segment]]
            actual = robot[indices[segment + 1]] - robot[indices[segment]]
            cosine = np.dot(desired, actual) / max(np.linalg.norm(desired) * np.linalg.norm(actual), 1e-12)
            direction_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    target_pinch = np.linalg.norm(target.target_points[4] - target.target_points[8])
    robot_pinch = np.linalg.norm(robot[4] - robot[8])
    return {
        "mean_point_error_cm": float(np.mean(position_errors) * 100.0),
        "tip_mean_error_cm": float(np.mean(position_errors[[4, 8, 12, 16, 20]]) * 100.0),
        "mean_segment_direction_error_deg": float(np.mean(direction_angles)),
        "thumb_index_distance_error_cm": float(abs(robot_pinch - target_pinch) * 100.0),
    }


__all__ = [
    "FINGER_CHAINS",
    "L25TargetChain",
    "build_l25_target_chain",
    "chain_error_metrics",
    "l25_chain_points",
]
