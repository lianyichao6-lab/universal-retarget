"""Validated AR5/L25 action conversion for the luban_framework interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .luban_contract import (
    AR5_RIGHT_JOINT_NAMES,
    LUBAN_RIGHT_ARM_ACTION,
    LUBAN_RIGHT_HAND_CONTROLLER,
    l25_active_joint_names,
    l25_qpos_to_luban_active,
)


@dataclass(frozen=True)
class LubanAction:
    """One synchronized position command for the right AR5 and L25."""

    arm_positions: np.ndarray
    hand_positions: np.ndarray
    time_from_start_s: float

    def __post_init__(self) -> None:
        arm = np.asarray(self.arm_positions, dtype=np.float64)
        hand = np.asarray(self.hand_positions, dtype=np.float64)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("arm_positions must be finite with shape (7,)")
        if hand.shape != (16,) or not np.isfinite(hand).all():
            raise ValueError("hand_positions must be finite with shape (16,)")
        if not np.isfinite(self.time_from_start_s) or self.time_from_start_s < 0:
            raise ValueError("time_from_start_s must be finite and non-negative")
        object.__setattr__(self, "arm_positions", arm.copy())
        object.__setattr__(self, "hand_positions", hand.copy())
        object.__setattr__(self, "time_from_start_s", float(self.time_from_start_s))


def build_luban_action(
    arm_positions: np.ndarray,
    l25_qpos: np.ndarray,
    *,
    time_from_start_s: float,
) -> LubanAction:
    """Convert one internal L25 qpos and one AR5 target to Luban units."""
    return LubanAction(
        arm_positions=np.asarray(arm_positions, dtype=np.float64),
        hand_positions=l25_qpos_to_luban_active(l25_qpos),
        time_from_start_s=time_from_start_s,
    )


def action_contract() -> dict[str, object]:
    """Return names/topics for logging and ROS message construction."""
    return {
        "arm_joint_names": AR5_RIGHT_JOINT_NAMES,
        "hand_joint_names": l25_active_joint_names(),
        "arm_action": LUBAN_RIGHT_ARM_ACTION,
        "hand_controller": LUBAN_RIGHT_HAND_CONTROLLER,
    }


__all__ = ["LubanAction", "action_contract", "build_luban_action"]
