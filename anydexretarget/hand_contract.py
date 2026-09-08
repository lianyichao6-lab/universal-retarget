"""Hand-model joint contracts independent of any robot middleware."""

from __future__ import annotations

import numpy as np

from .hardware_adapter import L25_QPOS_JOINTS


L25_ACTIVE_JOINT_NAMES = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
)
L25_ACTIVE_INDICES = np.asarray(
    [L25_QPOS_JOINTS.index(name) for name in L25_ACTIVE_JOINT_NAMES], dtype=np.int64
)

# Physical O30 URDF order returned by the Vector optimizer.
O30_QPOS_JOINT_NAMES = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_mcp", "thumb_ip",
)

# Vendor/controller semantic order. A transport-specific prefix is applied by
# that transport adapter, not by the retargeting core.
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
    [O30_QPOS_JOINT_NAMES.index(name) for name in O30_ACTIVE_SOURCE_NAMES], dtype=np.int64
)


def normalize_hand_model(hand_model: str) -> str:
    """Return the canonical local hand model name."""
    model = hand_model.lower().replace("linkerhand_", "")
    if model not in {"l25", "o30"}:
        raise ValueError("hand_model must be one of: l25, o30")
    return model


def l25_qpos_to_active(qpos: np.ndarray) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(L25_QPOS_JOINTS),) or not np.isfinite(values).all():
        raise ValueError(f"L25 qpos must be finite with shape ({len(L25_QPOS_JOINTS)},)")
    return values[L25_ACTIVE_INDICES].copy()


def o30_qpos_to_command_order(qpos: np.ndarray) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(O30_QPOS_JOINT_NAMES),) or not np.isfinite(values).all():
        raise ValueError(f"O30 qpos must be finite with shape ({len(O30_QPOS_JOINT_NAMES)},)")
    return values[O30_ACTIVE_INDICES].copy()


def hand_qpos_to_command_order(qpos: np.ndarray, *, hand_model: str) -> np.ndarray:
    return l25_qpos_to_active(qpos) if normalize_hand_model(hand_model) == "l25" else o30_qpos_to_command_order(qpos)


__all__ = [
    "L25_ACTIVE_JOINT_NAMES", "L25_ACTIVE_INDICES", "O30_QPOS_JOINT_NAMES",
    "O30_ACTIVE_JOINT_NAMES", "O30_ACTIVE_SOURCE_NAMES", "O30_ACTIVE_INDICES",
    "normalize_hand_model", "l25_qpos_to_active", "o30_qpos_to_command_order",
    "hand_qpos_to_command_order",
]
