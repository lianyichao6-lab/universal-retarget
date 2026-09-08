"""O30 Vector retargeting for HUG's standard 21-point hand output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand_contract import (
    O30_ACTIVE_JOINT_NAMES,
    O30_QPOS_JOINT_NAMES,
    o30_qpos_to_command_order,
)
from .o30_retarget_backend import retarget_o30_vector


@dataclass(frozen=True)
class HUGO30Result:
    """Static O30 target and the corresponding controller-order command."""

    qpos: np.ndarray
    qpos_joint_names: tuple[str, ...]
    command_positions: np.ndarray
    command_joint_names: tuple[str, ...]
    transformed_keypoints: np.ndarray
    cost: float


def retarget_hug_o30(keypoints: np.ndarray) -> HUGO30Result:
    """Convert one finite HUG 21x3 right-hand prediction to O30 qpos."""
    result = retarget_o30_vector(keypoints, hand_side="right")
    qpos = np.asarray(result.qpos, dtype=np.float32)
    if tuple(result.joint_names) != O30_QPOS_JOINT_NAMES:
        raise ValueError("O30 Vector output order does not match the audited O30 contract")
    command = o30_qpos_to_command_order(qpos).astype(np.float32)
    cost = float(result.geometry_retargeter.optimizer.compute_cost(
        qpos, result.transformed_keypoints
    ))
    return HUGO30Result(
        qpos=qpos,
        qpos_joint_names=O30_QPOS_JOINT_NAMES,
        command_positions=command,
        command_joint_names=O30_ACTIVE_JOINT_NAMES,
        transformed_keypoints=np.asarray(result.transformed_keypoints, dtype=np.float32),
        cost=cost,
    )


__all__ = ["HUGO30Result", "retarget_hug_o30"]
