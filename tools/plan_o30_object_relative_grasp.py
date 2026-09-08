#!/usr/bin/env python3
"""Fit an O30 Vector posture to HUG contacts on a reconstructed object mesh.

This mirrors the L25 object-relative planning stage. It is simulation-only.
"""
from __future__ import annotations

import argparse

import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.o30_retarget_backend import VECTOR_CONFIG
from anydexretarget.retarget import Retargeter

ROOT = Path(__file__).resolve().parents[1]
O30_MODEL = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation for target = scale * source @ R.T + t."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero / len(source))
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Cannot estimate similarity from degenerate hand keypoints")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Estimated non-positive O30 scale")
    return scale, rotation, target_center - scale * (rotation @ source_center)


def _transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * np.asarray(points, dtype=np.float64) @ rotation.T + translation[None]


def _load_contact_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {"source_canonical_grasp", "source_object_mesh", "object_to_camera", "surface_anchor_camera", "surface_normal_camera", "near_surface", "finger_names"}
    missing = required - set(result)
    if missing:
        raise ValueError("Contact plan missing: " + ", ".join(sorted(missing)))
    return result


def _joint_bounds(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    by_name = {str(name).lower(): index for index, name in enumerate(model_names) if name is not None}
    missing = [name for name in names if name.lower() not in by_name]
    if missing:
        raise ValueError("O30 MuJoCo model missing retarget joints: " + ", ".join(missing))
    indices = np.asarray([by_name[name.lower()] for name in names], dtype=np.int64)
    return model.jnt_range[indices, 0], model.jnt_range[indices, 1], [str(name) for name in model_names if name is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gap-mm", type=float, default=4.0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.10)
    parser.add_argument("--contact-scale-mm", type=float, default=10.0)
    parser.add_argument("--max-evaluations", type=int, default=240)
    args = parser.parse_args()
    if args.target_gap_mm < 0 or args.contact_weight <= 0 or args.posture_weight < 0 or args.contact_scale_mm <= 0 or args.max_evaluations <= 0:
        raise ValueError("Invalid optimization settings")

    contact = _load_contact_plan(args.contact_plan)
    state_path = Path(str(contact["source_canonical_grasp"].item()))
    state = load_canonical_grasp_state(state_path)
    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    baseline_q, verbose = retargeter.retarget_verbose(state.keypoints_for_retargeting(), apply_filter=False)
    robot = retargeter.optimizer.robot
    vector_names = [str(name) for name in robot.dof_joint_names]
    model = mujoco.MjModel.from_xml_path(str(O30_MODEL))
    lower, upper, model_names = _joint_bounds(model, vector_names)
    baseline_q = np.clip(np.asarray(baseline_q, dtype=np.float64), lower + 1e-6, upper - 1e-6)

    scale, rotation, translation = _similarity(state.keypoints_for_retargeting(), np.asarray(verbose["mediapipe_kp"], dtype=np.float64))
    anchors = np.asarray(contact["surface_anchor_camera"], dtype=np.float64)
    normals = np.asarray(contact["surface_normal_camera"], dtype=np.float64)
    active = np.asarray(contact["near_surface"], dtype=np.uint8).astype(bool)
    fingers = [str(item) for item in contact["finger_names"]]
    if anchors.shape != (5, 3) or normals.shape != (5, 3) or active.shape != (5,) or len(fingers) != 5:
        raise ValueError("Contact plan must have five fingertip anchors")
    if int(active.sum()) < 2:
        raise ValueError("Need at least two near-surface fingers for an O30 object-relative plan")
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    desired_o30 = _transform(anchors + normals * (args.target_gap_mm / 1000.0 / scale), scale, rotation, translation)

    task_names = [str(name) for name in retargeter.optimizer.task_link_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    if len(task_names) != 5 or task_offsets.shape != (5, 3):
        raise ValueError("O30 Vector config must expose five fingertip task points")
    tip_names = task_names
    tip_offsets = task_offsets
    task_ids = [robot.get_link_index(name) for name in tip_names]

    def fk(qpos: np.ndarray) -> np.ndarray:
        return robot.compute_points_batch(np.asarray(qpos, dtype=np.float64), task_ids, tip_offsets)

    baseline_tips = fk(baseline_q)
    ranges = np.maximum(upper - lower, 1e-6)
    contact_scale = args.contact_scale_mm / 1000.0

    def residual(qpos: np.ndarray) -> np.ndarray:
        return np.concatenate((
            np.sqrt(args.contact_weight) * (fk(qpos)[active] - desired_o30[active]).reshape(-1) / contact_scale,
            np.sqrt(args.posture_weight) * (qpos - baseline_q) / ranges,
        ))

    solve = least_squares(residual, baseline_q, bounds=(lower + 1e-6, upper - 1e-6), method="trf", max_nfev=args.max_evaluations)
    qpos = np.asarray(solve.x, dtype=np.float64)
    final_tips = fk(qpos)
    vector_by_name = {name.lower(): value for name, value in zip(vector_names, qpos)}
    qpos_model_order = np.asarray([vector_by_name[name.lower()] for name in model_names], dtype=np.float32)
    before = np.linalg.norm(baseline_tips - desired_o30, axis=1)
    after = np.linalg.norm(final_tips - desired_o30, axis=1)
    margins = np.minimum((qpos - lower) / ranges, (upper - qpos) / ranges)
    object_to_o30 = np.eye(4, dtype=np.float64)
    object_to_o30[:3, :3] = rotation
    object_to_o30[:3, 3] = _transform(np.asarray(contact["object_to_camera"], dtype=np.float64)[:3, 3][None], scale, rotation, translation)[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int64), simulation_only=np.asarray(True), robot=np.asarray("o30"),
        optimizer=np.asarray("object_aware_o30_vector_contact"), source_contact_plan=np.asarray(str(args.contact_plan.resolve())),
        source_canonical_grasp=np.asarray(str(state_path.resolve())), source_object_mesh=contact["source_object_mesh"],
        robot_joint_names=np.asarray(model_names), qpos=qpos_model_order, vector_joint_names=np.asarray(vector_names), qpos_vector_order=qpos.astype(np.float32),
        qpos_initial_retarget=baseline_q.astype(np.float32), active_contact_mask=active.astype(np.uint8), active_contact_fingers=np.asarray(fingers)[active],
        o30_fingertip_link_names=np.asarray(tip_names), o30_fingertip_task_offsets=tip_offsets.astype(np.float32),
        o30_fingertip_positions_baseline=baseline_tips.astype(np.float32), o30_fingertip_positions_optimized=final_tips.astype(np.float32),
        contact_target_positions_o30=desired_o30.astype(np.float32), fingertip_error_before_m=before.astype(np.float32), fingertip_error_after_m=after.astype(np.float32),
        camera_to_o30_rotation=rotation.astype(np.float32), camera_to_o30_translation=translation.astype(np.float32),
        human_to_o30_uniform_scale=np.asarray(scale, dtype=np.float32), object_to_o30=object_to_o30.astype(np.float32), object_uniform_scale_in_o30_frame=np.asarray(scale, dtype=np.float32),
        target_gap_mm=np.asarray(args.target_gap_mm, dtype=np.float32),
    )
    report = {
        "simulation_only": True, "hardware_command_generated": False, "source_contact_plan": str(args.contact_plan.resolve()),
        "method": "bounded_o30_fingertip_anchor_optimization_with_vector_posture_regularization",
        "active_contact_fingers": [name for name, enabled in zip(fingers, active) if enabled], "human_to_o30_uniform_scale": scale,
        "target_gap_mm": args.target_gap_mm, "optimizer_success": bool(solve.success), "function_evaluations": int(solve.nfev),
        "active_mean_error_before_mm": float(before[active].mean() * 1000.0), "active_mean_error_after_mm": float(after[active].mean() * 1000.0),
        "joint_saturation_count": int(np.count_nonzero(margins <= 0.05)),
        "note": "Object coordinates are valid in the O30 simulation hand frame only, not in an externally calibrated robot frame.",
    }
    trajectory_path = args.output.with_name(args.output.stem + "_trajectory.pkl")
    neutral = np.clip(np.zeros_like(qpos), lower + 1e-6, upper - 1e-6)
    with trajectory_path.open("wb") as stream:
        pickle.dump([{
            "target": (neutral + fraction * (qpos - neutral)).astype(np.float32),
            "sim_qpos": (neutral + fraction * (qpos - neutral)).astype(np.float32),
            "robot_joint_names": vector_names, "robot": "o30",
            "optimizer": "object_aware_o30_vector_contact",
        } for fraction in np.linspace(0.0, 1.0, 30)], stream)
    report["trajectory"] = str(trajectory_path.resolve())
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Object-aware O30 contact plan written (simulation only)")
    print(f"  active contacts: {', '.join(report['active_contact_fingers'])}")
    print(f"  active fingertip mean error: {report['active_mean_error_before_mm']:.2f} -> {report['active_mean_error_after_mm']:.2f} mm")
    print(f"  output: {args.output}")
    print(f"  trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
