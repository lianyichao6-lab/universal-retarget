"""Resolve an AnyDex grasp contract into Luban arm, hand and object requests."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .luban_arm import arm_flange_target, homogeneous_transform
from .luban_contract import l25_active_joint_names, l25_qpos_to_luban_active


LUBAN_GRASP_REQUEST_SCHEMA_VERSION = 2


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
    t_anchor_object: object | None = None,
    preshape_positions: object | None = None,
    lift_distance_m: float = 0.03,
    lift_direction_base: object = (0.0, 0.0, 1.0),
    object_cylinder_height_m: float = 0.068,
    object_cylinder_radius_m: float = 0.030,
) -> dict[str, np.ndarray]:
    """Build a fail-closed request using transforms captured with the RGB-D frame."""
    if not base_frame.strip():
        raise ValueError("base_frame cannot be empty")
    if not 0.0 < lift_distance_m <= 0.05:
        raise ValueError("lift_distance_m must be in (0, 0.05]")
    lift_direction = np.asarray(lift_direction_base, dtype=np.float64)
    if lift_direction.shape != (3,) or not np.isfinite(lift_direction).all() or np.linalg.norm(lift_direction) <= 1e-9:
        raise ValueError("lift_direction_base must be one finite non-zero 3-vector")
    lift_direction /= np.linalg.norm(lift_direction)
    if object_cylinder_height_m <= 0 or object_cylinder_radius_m <= 0:
        raise ValueError("object cylinder dimensions must be positive")

    anchor_frame = str(np.asarray(grasp_contract["anchor_frame"]).item())
    if expected_anchor_frame is not None and anchor_frame != expected_anchor_frame:
        raise ValueError(
            f"grasp plan anchor_frame={anchor_frame!r} does not match {expected_anchor_frame!r}"
        )
    qpos = np.asarray(grasp_contract["l25_qpos"], dtype=np.float64)
    active = l25_qpos_to_luban_active(qpos)
    preshape = np.zeros(16, dtype=np.float32) if preshape_positions is None else np.asarray(
        preshape_positions, dtype=np.float32
    )
    if preshape.shape != (16,) or not np.isfinite(preshape).all():
        raise ValueError("preshape_positions must contain 16 finite values")

    t_base_anchor = homogeneous_transform(
        t_robot_base_anchor_capture, "T_robot_base_anchor_capture"
    )
    t_flange_hand = homogeneous_transform(
        t_arm_flange_l25_hand, "T_arm_flange_l25_hand"
    )
    t_anchor_hand = homogeneous_transform(
        grasp_contract["T_anchor_l25_hand"], "T_anchor_l25_hand"
    )
    t_anchor_object_value = (
        np.eye(4, dtype=np.float64)
        if t_anchor_object is None
        else homogeneous_transform(t_anchor_object, "T_anchor_object")
    )
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
            grasp_contract["T_anchor_pregrasp_l25_hand"],
            "T_anchor_pregrasp_l25_hand",
        )
    else:
        offset = np.asarray(pregrasp_offset_hand_m, dtype=np.float64)
        t_anchor_pregrasp = t_anchor_hand @ _offset_transform(offset)
    pregrasp = homogeneous_transform(
        t_base_anchor @ t_anchor_pregrasp @ np.linalg.inv(t_flange_hand),
        "T_robot_base_arm_flange_pregrasp",
    )
    lift = target.copy()
    lift[:3, 3] += lift_direction * float(lift_distance_m)

    return {
        "schema_version": np.asarray(LUBAN_GRASP_REQUEST_SCHEMA_VERSION, dtype=np.int64),
        "planning_only": np.asarray(True),
        "hardware_ready": np.asarray(False),
        "base_frame": np.asarray(base_frame),
        "anchor_frame": np.asarray(anchor_frame),
        "candidate_id": np.asarray(str(np.asarray(grasp_contract.get("candidate_id", "")).item())),
        "T_robot_base_anchor_capture": t_base_anchor,
        "T_anchor_object": t_anchor_object_value,
        "T_robot_base_object": t_base_anchor @ t_anchor_object_value,
        "T_arm_flange_l25_hand": t_flange_hand,
        "T_robot_base_l25_hand_target": t_base_anchor @ t_anchor_hand,
        "T_robot_base_arm_flange_pregrasp": pregrasp,
        "T_robot_base_arm_flange_target": target,
        "T_robot_base_arm_flange_lift": lift,
        "pregrasp_offset_hand_m": offset.astype(np.float64),
        "lift_distance_m": np.asarray(lift_distance_m, dtype=np.float64),
        "lift_direction_base": lift_direction.astype(np.float64),
        "object_cylinder_height_m": np.asarray(object_cylinder_height_m, dtype=np.float64),
        "object_cylinder_radius_m": np.asarray(object_cylinder_radius_m, dtype=np.float64),
        "l25_qpos": qpos.astype(np.float32),
        "l25_preshape_positions": preshape,
        "l25_active_positions": active.astype(np.float32),
        "l25_active_joint_names": np.asarray(l25_active_joint_names()),
    }


__all__ = ["LUBAN_GRASP_REQUEST_SCHEMA_VERSION", "build_luban_grasp_request"]
