"""Joint-name contracts shared by AnyDexRetarget and luban_framework.

L25 planning in this repository uses 21 simulated joints. Luban's
ros2_control L25 controller exposes only the 16 actively commanded joints;
the distal joints are mechanically coupled in the hand hardware.
"""

from __future__ import annotations

import numpy as np

from .hardware_adapter import L25_QPOS_JOINTS


AR5_RIGHT_JOINT_NAMES = tuple(f"r_joint_{index}" for index in range(1, 8))
LUBAN_RIGHT_ARM_CONTROLLER = "/right_arm_controller"
LUBAN_RIGHT_ARM_ACTION = "/right_arm_controller/follow_joint_trajectory"
LUBAN_RIGHT_HAND_CONTROLLER = "/right_hand_controller/commands"
LUBAN_JOINT_STATES = "/joint_states"

L25_ACTIVE_JOINT_NAMES = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
)
L25_ACTIVE_INDICES = np.asarray(
    [L25_QPOS_JOINTS.index(name) for name in L25_ACTIVE_JOINT_NAMES],
    dtype=np.int64,
)


def l25_qpos_to_luban_active(qpos: np.ndarray) -> np.ndarray:
    """Extract the 16 active L25 radians expected by Luban's controller."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(L25_QPOS_JOINTS),):
        raise ValueError(f"L25 qpos must have shape ({len(L25_QPOS_JOINTS)},)")
    if not np.isfinite(values).all():
        raise ValueError("L25 qpos must contain only finite values")
    return values[L25_ACTIVE_INDICES].copy()


def l25_active_joint_names(*, side: str = "right") -> tuple[str, ...]:
    """Return Luban ros2_control names for one active L25 hand."""
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    return tuple(f"{side[0]}_hand_{name}" for name in L25_ACTIVE_JOINT_NAMES)


__all__ = [
    "AR5_RIGHT_JOINT_NAMES",
    "LUBAN_RIGHT_ARM_CONTROLLER",
    "LUBAN_RIGHT_ARM_ACTION",
    "LUBAN_RIGHT_HAND_CONTROLLER",
    "LUBAN_JOINT_STATES",
    "L25_ACTIVE_JOINT_NAMES",
    "L25_ACTIVE_INDICES",
    "l25_qpos_to_luban_active",
    "l25_active_joint_names",
]
