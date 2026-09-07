"""Validated AR5 plus LinkerHand L25/O30 action conversion for Luban."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .luban_contract import (
    AR5_RIGHT_JOINT_NAMES,
    LUBAN_RIGHT_ARM_ACTION,
    LUBAN_RIGHT_HAND_CONTROLLER,
    hand_active_joint_names,
    hand_qpos_to_luban_active,
    normalize_hand_model,
)


@dataclass(frozen=True)
class LubanAction:
    """One synchronized position command for the right AR5 and one hand."""

    arm_positions: np.ndarray
    hand_positions: np.ndarray
    time_from_start_s: float
    hand_model: str = "l25"

    def __post_init__(self) -> None:
        arm = np.asarray(self.arm_positions, dtype=np.float64)
        hand = np.asarray(self.hand_positions, dtype=np.float64)
        model = normalize_hand_model(self.hand_model)
        expected = len(hand_active_joint_names(hand_model=model))
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("arm_positions must be finite with shape (7,)")
        if hand.shape != (expected,) or not np.isfinite(hand).all():
            raise ValueError(f"{model.upper()} hand_positions must be finite with shape ({expected},)")
        if not np.isfinite(self.time_from_start_s) or self.time_from_start_s < 0:
            raise ValueError("time_from_start_s must be finite and non-negative")
        object.__setattr__(self, "arm_positions", arm.copy())
        object.__setattr__(self, "hand_positions", hand.copy())
        object.__setattr__(self, "time_from_start_s", float(self.time_from_start_s))
        object.__setattr__(self, "hand_model", model)


def build_luban_action(
    arm_positions: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    time_from_start_s: float,
    hand_model: str = "l25",
) -> LubanAction:
    """Convert one internal hand qpos and one AR5 target to Luban units."""
    model = normalize_hand_model(hand_model)
    return LubanAction(
        arm_positions=np.asarray(arm_positions, dtype=np.float64),
        hand_positions=hand_qpos_to_luban_active(hand_qpos, hand_model=model),
        time_from_start_s=time_from_start_s,
        hand_model=model,
    )


def action_contract(*, hand_model: str = "l25") -> dict[str, object]:
    """Return names/topics for logging and ROS message construction."""
    model = normalize_hand_model(hand_model)
    return {
        "arm_joint_names": AR5_RIGHT_JOINT_NAMES,
        "hand_joint_names": hand_active_joint_names(hand_model=model),
        "arm_action": LUBAN_RIGHT_ARM_ACTION,
        "hand_controller": LUBAN_RIGHT_HAND_CONTROLLER,
        "hand_model": model,
    }


__all__ = ["LubanAction", "action_contract", "build_luban_action"]
