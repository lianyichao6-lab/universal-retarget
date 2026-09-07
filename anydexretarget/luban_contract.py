"""Joint-name contracts shared by AnyDexRetarget and luban_framework.

L25 exposes 16 independently commanded joints from a 21-joint model. O30
uses the same physical geometry naming in AnyDex's URDF, but Luban's O30
ros2_control interface uses the 20 HOP semantic names. This module is the one
explicit conversion between those two layers.
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

# The O30 mesh URDF uses these physical joint names. This is exactly the order
# returned by the O30 KeyVectorOptimizer's Pinocchio model.
O30_QPOS_JOINT_NAMES = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_mcp", "thumb_ip",
)

# This is the direct Luban O30 controller and HOP transport order.
O30_ACTIVE_JOINT_NAMES = (
    "thumb_roll", "thumb_yaw", "index_yaw", "middle_yaw", "ring_yaw",
    "little_yaw", "thumb_root1", "index_root1", "middle_root1", "ring_root1",
    "little_root1", "index_root2", "middle_root2", "ring_root2", "little_root2",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
)
O30_ACTIVE_SOURCE_NAMES = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "index_mcp_roll", "middle_mcp_roll", "ring_mcp_roll",
    "pinky_mcp_roll", "thumb_mcp", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch",
    "pinky_mcp_pitch", "index_pip", "middle_pip", "ring_pip", "pinky_pip",
    "thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip",
)
O30_ACTIVE_INDICES = np.asarray(
    [O30_QPOS_JOINT_NAMES.index(name) for name in O30_ACTIVE_SOURCE_NAMES],
    dtype=np.int64,
)


def normalize_hand_model(hand_model: str) -> str:
    """Return the canonical local hand model name."""
    normalized = hand_model.lower().replace("linkerhand_", "")
    if normalized not in {"l25", "o30"}:
        raise ValueError("hand_model must be one of: l25, o30")
    return normalized


def l25_qpos_to_luban_active(qpos: np.ndarray) -> np.ndarray:
    """Extract the 16 active L25 radians expected by Luban's controller."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(L25_QPOS_JOINTS),):
        raise ValueError(f"L25 qpos must have shape ({len(L25_QPOS_JOINTS)},)")
    if not np.isfinite(values).all():
        raise ValueError("L25 qpos must contain only finite values")
    return values[L25_ACTIVE_INDICES].copy()


def o30_qpos_to_luban_active(qpos: np.ndarray) -> np.ndarray:
    """Convert O30 physical URDF qpos to Luban's direct HOP command order."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(O30_QPOS_JOINT_NAMES),):
        raise ValueError(f"O30 qpos must have shape ({len(O30_QPOS_JOINT_NAMES)},)")
    if not np.isfinite(values).all():
        raise ValueError("O30 qpos must contain only finite values")
    return values[O30_ACTIVE_INDICES].copy()


def hand_qpos_to_luban_active(qpos: np.ndarray, *, hand_model: str) -> np.ndarray:
    """Convert a model qpos to its right/left Luban command vector."""
    model = normalize_hand_model(hand_model)
    return l25_qpos_to_luban_active(qpos) if model == "l25" else o30_qpos_to_luban_active(qpos)


def hand_active_joint_names(*, hand_model: str, side: str = "right") -> tuple[str, ...]:
    """Return ros2_control joint names in controller command order."""
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    names = L25_ACTIVE_JOINT_NAMES if normalize_hand_model(hand_model) == "l25" else O30_ACTIVE_JOINT_NAMES
    return tuple(f"{side[0]}_hand_{name}" for name in names)


def l25_active_joint_names(*, side: str = "right") -> tuple[str, ...]:
    """Backward-compatible L25 controller name helper."""
    return hand_active_joint_names(hand_model="l25", side=side)


def o30_active_joint_names(*, side: str = "right") -> tuple[str, ...]:
    """Return the 20 O30 ros2_control names in HOP order."""
    return hand_active_joint_names(hand_model="o30", side=side)


__all__ = [
    "AR5_RIGHT_JOINT_NAMES", "LUBAN_RIGHT_ARM_CONTROLLER", "LUBAN_RIGHT_ARM_ACTION",
    "LUBAN_RIGHT_HAND_CONTROLLER", "LUBAN_JOINT_STATES", "L25_ACTIVE_JOINT_NAMES",
    "L25_ACTIVE_INDICES", "O30_QPOS_JOINT_NAMES", "O30_ACTIVE_JOINT_NAMES",
    "O30_ACTIVE_SOURCE_NAMES", "O30_ACTIVE_INDICES", "normalize_hand_model",
    "l25_qpos_to_luban_active", "o30_qpos_to_luban_active", "hand_qpos_to_luban_active",
    "hand_active_joint_names", "l25_active_joint_names", "o30_active_joint_names",
]
