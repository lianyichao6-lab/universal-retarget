"""Resolve an AnyDex grasp contract into Luban arm and hand requests."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .luban_arm import arm_flange_target, homogeneous_transform
from .luban_contract import l25_active_joint_names, l25_qpos_to_luban_active


LUBAN_GRASP_REQUEST_SCHEMA_VERSION = 1


def _offset_transform(offset: object) -> np.ndarray:
    value = np.asarray(offset, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("pregrasp_offset_hand_m must be finite with shape (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = value
    return transform


def build_luban_grasp_request(
    grasp_contract: Mapping[str, object],
    *,
    t_robot_base_anchor_capture: object,
    t_arm_flange_l25_hand: object,
    base_frame: str,
    expected_anchor_frame: str | None = None,
    pregrasp_offset_hand_m: object | None = None,
) -> dict[str, np.ndarray]:
    """Build a request using the robot-to-anchor transform at capture time."""
    if not base_frame.strip():
        raise ValueError("base_frame cannot be empty")
    anchor_frame = str(np.asarray(grasp_contract["anchor_frame"]).item())
    if expected_anchor_frame is not None and anchor_frame != expected_anchor_frame:
        raise ValueError(
            f"grasp plan anchor_frame={anchor_frame!r} does not match {expected_anchor_frame!r}"
        )
    qpos = np.asarray(grasp_contract["l25_qpos"], dtype=np.float64)
    active = l25_qpos_to_luban_active(qpos)
    t_base_anchor = homogeneous_transform(
        t_robot_base_anchor_capture, "T_robot_base_anchor_capture"
    )
    t_flange_hand = homogeneous_transform(t_arm_flange_l25_hand, "T_arm_flange_l25_hand")
    t_anchor_hand = homogeneous_transform(grasp_contract["T_anchor_l25_hand"], "T_anchor_l25_hand")
    target = arm_flange_target(
        grasp_contract,
        t_robot_base_anchor=t_base_anchor,
        t_arm_flange_l25_hand=t_flange_hand,
    )
    if pregrasp_offset_hand_m is None:
        if not bool(np.asarray(grasp_contract.get("pregrasp_defined", False)).item()):
            raise ValueError("A validated pregrasp offset is required")
        offset = np.asarray(grasp_contract["pregrasp_offset_hand_m"], dtype=np.float64)
        t_anchor_pregrasp = homogeneous_transform(
            grasp_contract["T_anchor_pregrasp_l25_hand"], "T_anchor_pregrasp_l25_hand"
        )
    else:
        offset = np.asarray(pregrasp_offset_hand_m, dtype=np.float64)
        t_anchor_pregrasp = t_anchor_hand @ _offset_transform(offset)
    pregrasp = arm_flange_target(
        {"T_anchor_pregrasp_l25_hand": t_anchor_pregrasp},
        t_robot_base_anchor=t_base_anchor,
        t_arm_flange_l25_hand=t_flange_hand,
        hand_pose_key="T_anchor_pregrasp_l25_hand",
    )
    return {
        "schema_version": np.asarray(LUBAN_GRASP_REQUEST_SCHEMA_VERSION, dtype=np.int64),
        "planning_only": np.asarray(True),
        "hardware_ready": np.asarray(False),
        "base_frame": np.asarray(base_frame),
        "anchor_frame": np.asarray(anchor_frame),
        "candidate_id": np.asarray(str(np.asarray(grasp_contract.get("candidate_id", "")).item())),
        "T_robot_base_anchor_capture": t_base_anchor,
        "T_arm_flange_l25_hand": t_flange_hand,
        "T_robot_base_l25_hand_target": t_base_anchor @ t_anchor_hand,
        "T_robot_base_arm_flange_pregrasp": pregrasp,
        "T_robot_base_arm_flange_target": target,
        "pregrasp_offset_hand_m": offset.astype(np.float64),
        "l25_qpos": qpos.astype(np.float32),
        "l25_active_positions": active.astype(np.float32),
        "l25_active_joint_names": np.asarray(l25_active_joint_names()),
    }


__all__ = ["LUBAN_GRASP_REQUEST_SCHEMA_VERSION", "build_luban_grasp_request"]
