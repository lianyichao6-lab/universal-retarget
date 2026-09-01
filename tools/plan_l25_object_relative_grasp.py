#!/usr/bin/env python3
"""Plan an object-relative static L25 grasp from HUG surface anchors.

This is an offline simulation-frame optimizer.  It preserves the current
AnyDex human-to-L25 similarity transform, maps the HUG object anchors through
that transform, and refines only L25 joint angles toward those anchors.  It
does not establish a camera-to-real-robot calibration or command hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"
L25_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation for target = scale * source @ R.T + t."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must both be N x 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Cannot estimate similarity from degenerate hand keypoints")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = target_center - scale * (rotation @ source_center)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Estimated non-positive/invalid human-to-L25 scale")
    return scale, rotation, translation


def _transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * np.asarray(points, dtype=np.float64) @ rotation.T + translation[None]


def _load_plan(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]).copy() for key in data.files}
    required = (
        "source_canonical_grasp", "source_object_mesh", "object_to_camera",
        "surface_anchor_camera", "surface_anchor_object", "surface_normal_camera",
        "near_surface", "tip_surface_distance_m", "finger_names",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Contact plan is missing fields: {missing}")
    return result


def _joint_bounds(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    if any(name is None for name in model_names):
        raise ValueError("L25 MuJoCo model has unnamed joints")
    by_name = {str(name).lower(): index for index, name in enumerate(model_names)}
    missing = [name for name in names if name.lower() not in by_name]
    if missing:
        raise ValueError(f"L25 MuJoCo model missing retarget joints: {missing}")
    indices = np.asarray([by_name[name.lower()] for name in names], dtype=np.int64)
    return model.jnt_range[indices, 0].astype(np.float64), model.jnt_range[indices, 1].astype(np.float64), [str(name) for name in model_names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vector-config", type=Path, default=VECTOR_CONFIG)
    parser.add_argument("--target-gap-mm", type=float, default=4.0,
                        help="Desired robot fingertip clearance along the reconstructed outward normal.")
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.10,
                        help="Preserve the baseline AnyDex L25 posture.")
    parser.add_argument("--contact-scale-mm", type=float, default=10.0)
    parser.add_argument("--max-evaluations", type=int, default=240)
    args = parser.parse_args()
    if (args.target_gap_mm < 0 or args.contact_weight <= 0 or args.posture_weight < 0
            or args.contact_scale_mm <= 0 or args.max_evaluations <= 0):
        raise ValueError("Invalid contact optimization weights or limits")

    plan = _load_plan(args.contact_plan)
    canonical_path = Path(str(plan["source_canonical_grasp"].item()))
    state = load_canonical_grasp_state(canonical_path)
    retargeter = Retargeter.from_yaml(str(args.vector_config), hand_side="right")
    baseline_q, verbose = retargeter.retarget_verbose(state.keypoints_for_retargeting(), apply_filter=False)
    robot = retargeter.optimizer.robot
    joint_names = [str(name) for name in robot.dof_joint_names]
    baseline_q = np.asarray(baseline_q, dtype=np.float64)
    if baseline_q.shape != (len(joint_names),) or not np.isfinite(baseline_q).all():
        raise ValueError("AnyDex baseline qpos is invalid")
    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    lower, upper, model_joint_names = _joint_bounds(model, joint_names)
    baseline_q = np.clip(baseline_q, lower + 1e-6, upper - 1e-6)

    transformed_hand = np.asarray(verbose["mediapipe_kp"], dtype=np.float64)
    source_hand = state.keypoints_for_retargeting().astype(np.float64)
    scale, rotation, translation = _similarity(source_hand, transformed_hand)
    anchors_camera = np.asarray(plan["surface_anchor_camera"], dtype=np.float64)
    normals_camera = np.asarray(plan["surface_normal_camera"], dtype=np.float64)
    active = np.asarray(plan["near_surface"], dtype=np.uint8).astype(bool)
    names = [str(value) for value in plan["finger_names"]]
    if anchors_camera.shape != (5, 3) or normals_camera.shape != (5, 3) or active.shape != (5,):
        raise ValueError("Contact plan has invalid fingertip anchor arrays")
    if active.sum() < 2:
        raise ValueError("Need at least two near-surface fingertips for an object-relative L25 plan")
    normals_norm = np.linalg.norm(normals_camera, axis=1, keepdims=True)
    normals_camera = normals_camera / np.maximum(normals_norm, 1e-12)
    gap_camera = args.target_gap_mm / 1000.0 / scale
    desired_camera = anchors_camera + normals_camera * gap_camera
    desired_l25 = _transform(desired_camera, scale, rotation, translation)
    object_origin_camera = np.asarray(plan["object_to_camera"], dtype=np.float64)[:3, 3]
    object_translation_l25 = _transform(object_origin_camera[None], scale, rotation, translation)[0]

    task_names = list(retargeter.optimizer.task_link_names)
    task_ids = [robot.get_link_index(name) for name in task_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    if len(task_ids) != 5 or task_offsets.shape != (5, 3):
        raise ValueError("L25 Vector config does not expose five audited fingertip task points")

    def fingertip_fk(qpos: np.ndarray) -> np.ndarray:
        return robot.compute_points_batch(np.asarray(qpos, dtype=np.float64), task_ids, task_offsets)

    baseline_tips = fingertip_fk(baseline_q)
    ranges = np.maximum(upper - lower, 1e-6)
    contact_scale = args.contact_scale_mm / 1000.0

    def residual(qpos: np.ndarray) -> np.ndarray:
        tips = fingertip_fk(qpos)
        contact = (tips[active] - desired_l25[active]).reshape(-1) / contact_scale
        posture = (qpos - baseline_q) / ranges
        return np.concatenate((
            np.sqrt(args.contact_weight) * contact,
            np.sqrt(args.posture_weight) * posture,
        ))

    result = least_squares(
        residual,
        baseline_q,
        bounds=(lower + 1e-6, upper - 1e-6),
        max_nfev=args.max_evaluations,
        method="trf",
    )
    qpos = np.asarray(result.x, dtype=np.float64)
    final_tips = fingertip_fk(qpos)
    if not np.isfinite(qpos).all() or not np.isfinite(final_tips).all():
        raise RuntimeError("Object-aware L25 solve produced NaN/Inf")
    source_by_name = {name.lower(): index for index, name in enumerate(joint_names)}
    qpos_model_order = np.asarray(
        [qpos[source_by_name[name.lower()]] for name in model_joint_names], dtype=np.float32
    )
    baseline_errors = np.linalg.norm(baseline_tips - desired_l25, axis=1)
    final_errors = np.linalg.norm(final_tips - desired_l25, axis=1)
    margin = np.minimum((qpos - lower) / ranges, (upper - qpos) / ranges)

    object_to_l25 = np.eye(4, dtype=np.float64)
    object_to_l25[:3, :3] = rotation
    object_to_l25[:3, 3] = object_translation_l25
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int64),
        simulation_only=np.asarray(True),
        source_contact_plan=np.asarray(str(args.contact_plan.resolve())),
        source_canonical_grasp=np.asarray(str(canonical_path.resolve())),
        robot=np.asarray("l25"),
        optimizer=np.asarray("object_aware_l25_vector_contact"),
        robot_joint_names=np.asarray(model_joint_names),
        qpos=qpos_model_order,
        qpos_vector_order=qpos.astype(np.float32),
        vector_joint_names=np.asarray(joint_names),
        active_contact_fingers=np.asarray(names)[active],
        active_contact_mask=active.astype(np.uint8),
        l25_fingertip_positions_baseline=baseline_tips.astype(np.float32),
        l25_fingertip_positions_optimized=final_tips.astype(np.float32),
        desired_fingertip_positions_l25=desired_l25.astype(np.float32),
        fingertip_error_before_m=baseline_errors.astype(np.float32),
        fingertip_error_after_m=final_errors.astype(np.float32),
        camera_to_l25_rotation=rotation.astype(np.float32),
        camera_to_l25_translation=translation.astype(np.float32),
        human_to_l25_uniform_scale=np.asarray(scale, dtype=np.float32),
        object_to_l25=object_to_l25.astype(np.float32),
        object_uniform_scale_in_l25_frame=np.asarray(scale, dtype=np.float32),
        target_gap_mm=np.asarray(args.target_gap_mm, dtype=np.float32),
    )
    report = {
        "simulation_only": True,
        "source_contact_plan": str(args.contact_plan.resolve()),
        "source_canonical_grasp": str(canonical_path.resolve()),
        "method": "bounded_l25_fingertip_anchor_optimization_with_anydex_posture_regularization",
        "active_contact_fingers": [name for name, is_active in zip(names, active) if is_active],
        "human_to_l25_uniform_scale": scale,
        "object_frame_note": "Object pose is derived through AnyDex's hand similarity transform. It is valid only in the L25 simulation target frame, not an externally calibrated real robot frame.",
        "target_gap_mm": args.target_gap_mm,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "fingertip_error_before_mm": {name: float(error * 1000.0) for name, error in zip(names, baseline_errors)},
        "fingertip_error_after_mm": {name: float(error * 1000.0) for name, error in zip(names, final_errors)},
        "active_mean_error_before_mm": float(np.mean(baseline_errors[active]) * 1000.0),
        "active_mean_error_after_mm": float(np.mean(final_errors[active]) * 1000.0),
        "joint_limit_violations": int(np.count_nonzero((qpos < lower) | (qpos > upper))),
        "joint_saturation_count": int(np.count_nonzero(margin <= 0.05)),
        "collision_checked": False,
        "hardware_command_generated": False,
        "note": "This plan matches selected L25 fingertip targets to HUG-derived mesh anchors. It does not yet test full-link mesh collisions, force closure, arm approach, camera-to-robot extrinsics, or physical stability.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Object-aware L25 contact plan written (simulation only)")
    print(f"  active contacts: {', '.join(report['active_contact_fingers'])}")
    print(f"  active fingertip mean error: {report['active_mean_error_before_mm']:.2f} -> {report['active_mean_error_after_mm']:.2f} mm")
    print(f"  joint saturation: {report['joint_saturation_count']}; limit violations: {report['joint_limit_violations']}")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
