"""Portable deployment contracts for reconstruction and robot grasp plans."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


RECONSTRUCTION_SCHEMA_VERSION = 1
GRASP_EXECUTION_SCHEMA_VERSION = 1


def _array(
    values: object,
    shape: tuple[int, ...],
    name: str,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


def rigid_transform(rotation: object, translation: object) -> np.ndarray:
    """Return a validated homogeneous target-from-source rigid transform."""
    rotation = _array(rotation, (3, 3), "rotation")
    translation = _array(translation, (3,), "translation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("rotation must be proper with determinant +1")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def build_grasp_execution_plan(
    final_plan: Mapping[str, object],
    *,
    source_plan: str | Path,
    anchor_frame: str,
    hand_side: str = "right",
    candidate_id: str = "",
    object_mesh: str | Path | None = None,
    reconstruction_result: str | Path | None = None,
    pregrasp_offset_hand_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convert a simulation L25 plan into a calibration-ready robot contract.

    Existing plans store camera points mapped into the L25 FK frame. Therefore
    T_l25_hand_anchor is read directly and inverted to obtain the desired L25
    hand-base pose T_anchor_l25_hand.
    """
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be left or right")
    if not anchor_frame.strip():
        raise ValueError("anchor_frame cannot be empty")
    required = {
        "qpos",
        "robot_joint_names",
        "camera_to_l25_rotation",
        "camera_to_l25_translation",
        "retarget_backend",
    }
    missing = required - set(final_plan)
    if missing:
        raise ValueError("Final L25 plan missing: " + ", ".join(sorted(missing)))

    qpos = _array(final_plan["qpos"], (21,), "qpos", np.float32)
    joint_names = np.asarray(final_plan["robot_joint_names"])
    if joint_names.shape != (21,):
        raise ValueError("robot_joint_names must have shape (21,)")

    t_l25_anchor = rigid_transform(
        final_plan["camera_to_l25_rotation"],
        final_plan["camera_to_l25_translation"],
    )
    t_anchor_l25 = np.linalg.inv(t_l25_anchor)
    pregrasp_defined = pregrasp_offset_hand_m is not None
    if pregrasp_defined:
        offset = _array(pregrasp_offset_hand_m, (3,), "pregrasp_offset_hand_m")
        local_offset = np.eye(4, dtype=np.float64)
        local_offset[:3, 3] = offset
        t_anchor_pregrasp = t_anchor_l25 @ local_offset
    else:
        offset = np.full(3, np.nan, dtype=np.float64)
        t_anchor_pregrasp = np.full((4, 4), np.nan, dtype=np.float64)

    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(GRASP_EXECUTION_SCHEMA_VERSION, dtype=np.int64),
        "planning_only": np.asarray(True),
        "hardware_ready": np.asarray(False),
        "requires_target_machine_calibration": np.asarray(True),
        "source_final_l25_plan": np.asarray(str(Path(source_plan).resolve())),
        "source_reconstruction_result": np.asarray(
            "" if reconstruction_result is None else str(Path(reconstruction_result))
        ),
        "source_object_mesh": np.asarray(
            "" if object_mesh is None else str(Path(object_mesh))
        ),
        "candidate_id": np.asarray(candidate_id),
        "robot": np.asarray("l25"),
        "hand_side": np.asarray(hand_side),
        "retarget_backend": np.asarray(str(np.asarray(final_plan["retarget_backend"]).item())),
        "anchor_frame": np.asarray(anchor_frame),
        "transform_convention": np.asarray("T_target_source maps source-frame points into target"),
        "T_l25_hand_anchor": t_l25_anchor.astype(np.float64),
        "T_anchor_l25_hand": t_anchor_l25.astype(np.float64),
        "pregrasp_defined": np.asarray(pregrasp_defined),
        "pregrasp_offset_hand_m": offset,
        "T_anchor_pregrasp_l25_hand": t_anchor_pregrasp,
        "l25_joint_names": joint_names.copy(),
        "l25_qpos": qpos,
        "required_target_transforms": np.asarray(
            ("T_robot_base_anchor", "T_arm_flange_l25_hand")
        ),
        "execution_formula": np.asarray(
            "T_robot_base_arm_flange = "
            "T_robot_base_anchor @ T_anchor_l25_hand @ inv(T_arm_flange_l25_hand)"
        ),
    }
    for key in (
        "active_contact_mask",
        "contact_error_after_m",
        "object_penetration_after_m",
        "self_penetration_after_m",
    ):
        if key in final_plan:
            result[key] = np.asarray(final_plan[key]).copy()
    return result


def build_object_anchored_grasp_execution_plan(final_plan: Mapping[str, object], contact_plan: Mapping[str, object], *, source_plan: str | Path, source_contact_plan: str | Path, anchor_frame: str, hand_side: str = 'right', candidate_id: str = '') -> dict[str, np.ndarray]:
    "Build a simulation-only contract anchored in the HUG object frame."
    if 'object_to_camera' not in contact_plan:
        raise ValueError('Contact plan is missing object_to_camera')
    object_raw = np.asarray(contact_plan['object_to_camera'], dtype=np.float64)
    if object_raw.shape != (4, 4):
        raise ValueError('T_camera_object must have shape (4, 4)')
    object_to_camera = rigid_transform(object_raw[:3, :3], object_raw[:3, 3])
    camera_to_l25 = rigid_transform(final_plan['camera_to_l25_rotation'], final_plan['camera_to_l25_translation'])
    l25_to_object = camera_to_l25 @ object_to_camera
    anchored = dict(final_plan)
    anchored['camera_to_l25_rotation'] = l25_to_object[:3, :3]
    anchored['camera_to_l25_translation'] = l25_to_object[:3, 3]
    mesh = contact_plan.get('source_object_mesh')
    if mesh is None:
        raise ValueError('Contact plan is missing source_object_mesh')
    result = build_grasp_execution_plan(anchored, source_plan=source_plan, anchor_frame=anchor_frame, hand_side=hand_side, candidate_id=candidate_id, object_mesh=str(np.asarray(mesh).item()))
    result['source_contact_plan'] = np.asarray(str(Path(source_contact_plan).resolve()))
    result['anchor_kind'] = np.asarray('hug_object_frame')
    result['T_l25_hand_camera'] = camera_to_l25.astype(np.float64)
    result['T_camera_anchor'] = object_to_camera.astype(np.float64)
    result['T_l25_hand_anchor'] = l25_to_object.astype(np.float64)
    result['T_anchor_l25_hand'] = np.linalg.inv(l25_to_object).astype(np.float64)
    return result


__all__ = [
    "GRASP_EXECUTION_SCHEMA_VERSION",
    "RECONSTRUCTION_SCHEMA_VERSION",
    "build_object_anchored_grasp_execution_plan",
    "build_grasp_execution_plan",
    "rigid_transform",
]
