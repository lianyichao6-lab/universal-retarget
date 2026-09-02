"""AR5 flange target computation for the Luban deployment contract."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .deployment import rigid_transform


def homogeneous_transform(value: object, name: str) -> np.ndarray:
    """Validate a finite target-from-source homogeneous rigid transform."""
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be finite with shape (4, 4), got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rigid_transform(transform[:3, :3], transform[:3, 3])
    return transform.copy()


def arm_flange_target(
    grasp_contract: Mapping[str, object],
    *,
    t_robot_base_anchor: object,
    t_arm_flange_l25_hand: object,
) -> np.ndarray:
    """Compute ``T_robot_base_arm_flange`` from a grasp contract and calibration."""
    if "T_anchor_l25_hand" not in grasp_contract:
        raise ValueError("grasp contract is missing T_anchor_l25_hand")
    t_anchor_l25_hand = homogeneous_transform(
        grasp_contract["T_anchor_l25_hand"], "T_anchor_l25_hand"
    )
    t_base_anchor = homogeneous_transform(t_robot_base_anchor, "T_robot_base_anchor")
    t_flange_hand = homogeneous_transform(t_arm_flange_l25_hand, "T_arm_flange_l25_hand")
    target = t_base_anchor @ t_anchor_l25_hand @ np.linalg.inv(t_flange_hand)
    return homogeneous_transform(target, "T_robot_base_arm_flange")


def rotation_matrix_to_quaternion_xyzw(rotation: object) -> np.ndarray:
    """Convert a proper rotation matrix to a ROS ``[x, y, z, w]`` quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    rigid_transform(matrix, np.zeros(3),)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
            quaternion = np.array(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
            quaternion = np.array(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
            quaternion = np.array(
                [(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale,
                 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def arm_flange_pose_xyzw(target: object) -> tuple[np.ndarray, np.ndarray]:
    """Return translation and ROS-order quaternion for a flange transform."""
    transform = homogeneous_transform(target, "T_robot_base_arm_flange")
    return transform[:3, 3].copy(), rotation_matrix_to_quaternion_xyzw(transform[:3, :3])


__all__ = [
    "arm_flange_pose_xyzw",
    "arm_flange_target",
    "homogeneous_transform",
    "rotation_matrix_to_quaternion_xyzw",
]
