#!/usr/bin/env python3
"""Refine an L25 object-relative plan against MuJoCo mesh penetration.

The input plan remains the source of the intended fingertip contacts.  This
tool only adjusts joints in finger chains that collide with the reconstructed
object proxy, balancing target error, posture change and contact penetration.
It is an offline MuJoCo optimization and never communicates with hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize

from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ROOT_OFFSET = np.asarray((0.0, 0.0, 0.05), dtype=np.float64)


def _load_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        plan = {key: np.asarray(source[key]).copy() for key in source.files}
    required = {
        "qpos", "robot_joint_names", "qpos_vector_order", "vector_joint_names",
        "contact_target_positions_l25", "active_contact_mask",
    }
    missing = required - set(plan)
    if missing:
        raise ValueError("Plan missing: " + ", ".join(sorted(missing)))
    return plan


def _joint_map(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result[str(name).lower()] = joint_id
    return result


def _set_q(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray,
           vector_names: list[str], joint_ids: dict[str, int]) -> None:
    data.qpos[:] = 0.0
    for value, name in zip(q, vector_names):
        joint_id = joint_ids[name.lower()]
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _contacts(model: mujoco.MjModel, data: mujoco.MjData, object_id: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if object_id not in (contact.geom1, contact.geom2):
            continue
        hand_id = contact.geom2 if contact.geom1 == object_id else contact.geom1
        hand_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_id) or f"geom_{hand_id}"
        result.append({
            "hand_geom": str(hand_name),
            "distance_m": float(contact.dist),
            "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
        })
    return result


def _collision_fingers(contacts: list[dict[str, object]]) -> set[str]:
    result: set[str] = set()
    for contact in contacts:
        name = str(contact["hand_geom"]).lower()
        for finger in FINGERS:
            if name.startswith(finger + "_"):
                result.add(finger)
    return result


def _tip_points(model: mujoco.MjModel, data: mujoco.MjData, task_names: list[str],
                offsets: np.ndarray) -> np.ndarray:
    points = []
    for name, offset in zip(task_names, offsets):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model lacks task body: {name}")
        points.append(data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset - ROOT_OFFSET)
    return np.asarray(points, dtype=np.float64)


def _normalised_margins(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.maximum(upper - lower, 1e-6)
    return np.minimum((q - lower) / span, (upper - q) / span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Existing L25 object-relative plan.")
    parser.add_argument("--scene-xml", type=Path, required=True, help="MuJoCo scene built from that plan.")
    parser.add_argument("--output", type=Path, required=True, help="New collision-aware plan; source remains unchanged.")
    parser.add_argument("--vector-config", type=Path, default=VECTOR_CONFIG)
    parser.add_argument("--collision-weight", type=float, default=22.0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.08)
    parser.add_argument("--limit-weight", type=float, default=0.04)
    parser.add_argument("--contact-scale-mm", type=float, default=8.0)
    parser.add_argument("--penetration-scale-mm", type=float, default=2.0)
    parser.add_argument("--minimum-margin", type=float, default=0.015)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.10,
                        help="Conservative maximum deviation from the source plan for an optimized joint.")
    parser.add_argument("--max-iterations", type=int, default=160)
    parser.add_argument("--global-search", action="store_true", help="Use a slower global seed search before local refinement.")
    args = parser.parse_args()
    if not args.plan.is_file() or not args.scene_xml.is_file():
        raise FileNotFoundError("--plan and --scene-xml must both exist")
    if min(args.collision_weight, args.contact_weight, args.contact_scale_mm,
           args.penetration_scale_mm, args.max_iterations, args.max_joint_delta_rad) <= 0 or args.posture_weight < 0:
        raise ValueError("Invalid optimization weights or iteration limit")

    plan = _load_plan(args.plan)
    vector_names = [str(value) for value in plan["vector_joint_names"]]
    q_initial = np.asarray(plan["qpos_vector_order"], dtype=np.float64)
    targets = np.asarray(plan["contact_target_positions_l25"], dtype=np.float64)
    active = np.asarray(plan["active_contact_mask"], dtype=np.uint8).astype(bool)
    if q_initial.shape != (len(vector_names),) or targets.shape != (5, 3) or active.shape != (5,):
        raise ValueError("Unexpected L25 plan dimensions")

    model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    data = mujoco.MjData(model)
    joint_ids = _joint_map(model)
    if any(name.lower() not in joint_ids for name in vector_names):
        raise ValueError("Scene joint names disagree with plan vector joint order")
    vector_joint_ids = np.asarray([joint_ids[name.lower()] for name in vector_names], dtype=np.int64)
    lower = model.jnt_range[vector_joint_ids, 0].astype(np.float64)
    upper = model.jnt_range[vector_joint_ids, 1].astype(np.float64)
    q_initial = np.clip(q_initial, lower + 1e-6, upper - 1e-6)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reconstructed_object_geom")
    if object_id < 0:
        raise ValueError("Scene has no reconstructed_object_geom")

    retargeter = Retargeter.from_yaml(str(args.vector_config), hand_side="right")
    task_names = [str(value) for value in retargeter.optimizer.task_link_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    if task_offsets.shape != (5, 3):
        raise ValueError("Expected five L25 Vector task offsets")

    _set_q(model, data, q_initial, vector_names, joint_ids)
    contacts_before = _contacts(model, data, object_id)
    fingers = _collision_fingers(contacts_before)
    if not fingers:
        fingers = {FINGERS[index] for index, is_active in enumerate(active) if is_active}
    variable_indices = np.asarray([
        index for index, name in enumerate(vector_names)
        if any(name.lower().startswith(finger + "_") for finger in fingers)
    ], dtype=np.int64)
    if len(variable_indices) == 0:
        raise ValueError("Could not identify L25 joints for colliding fingers")
    ranges = np.maximum(upper - lower, 1e-6)
    contact_scale = args.contact_scale_mm / 1000.0
    penetration_scale = args.penetration_scale_mm / 1000.0

    def evaluate(partial: np.ndarray, detailed: bool = False):
        q = q_initial.copy()
        q[variable_indices] = np.asarray(partial, dtype=np.float64)
        _set_q(model, data, q, vector_names, joint_ids)
        tips = _tip_points(model, data, task_names, task_offsets)
        contacts = _contacts(model, data, object_id)
        penetrations = np.asarray([max(0.0, -float(item["distance_m"])) for item in contacts], dtype=np.float64)
        contact_error = (tips[active] - targets[active]).reshape(-1) / contact_scale
        posture_error = (q - q_initial) / ranges
        margin_penalty = np.maximum(0.0, args.minimum_margin - _normalised_margins(q, lower, upper))
        cost = (
            args.contact_weight * float(np.dot(contact_error, contact_error))
            + args.posture_weight * float(np.dot(posture_error, posture_error))
            + args.collision_weight * float(np.sum((penetrations / penetration_scale) ** 2))
            + args.limit_weight * float(np.dot(margin_penalty, margin_penalty))
        )
        if detailed:
            return cost, q, tips, contacts, penetrations
        return cost

    bounds = [
        (float(max(lower[index] + 1e-6, q_initial[index] - args.max_joint_delta_rad)),
         float(min(upper[index] - 1e-6, q_initial[index] + args.max_joint_delta_rad)))
        for index in variable_indices
    ]
    start = q_initial[variable_indices]
    if args.global_search:
        global_result = differential_evolution(evaluate, bounds=bounds, seed=7, polish=False,
                                               maxiter=max(8, args.max_iterations // 4), workers=1)
        start = np.asarray(global_result.x, dtype=np.float64)
    result = minimize(evaluate, start, method="Powell", bounds=bounds,
                      options={"maxiter": args.max_iterations, "xtol": 1e-4, "ftol": 1e-6})
    _cost_before, _q_before, tips_before, _before_contacts, before_penetrations = evaluate(q_initial[variable_indices], detailed=True)
    final_cost, q_final, tips_final, contacts_after, after_penetrations = evaluate(result.x, detailed=True)
    q_model_names = [str(value) for value in plan["robot_joint_names"]]
    vector_by_name = {name.lower(): value for name, value in zip(vector_names, q_final)}
    q_model = np.asarray([vector_by_name[name.lower()] for name in q_model_names], dtype=np.float32)
    error_before = np.linalg.norm(tips_before - targets, axis=1)
    error_after = np.linalg.norm(tips_final - targets, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.update({
        "schema_version": np.asarray(3, dtype=np.int64),
        "optimizer": np.asarray("l25_collision_aware_object_relative"),
        "source_l25_object_relative_plan": np.asarray(str(args.plan.resolve())),
        "qpos": q_model,
        "qpos_vector_order": q_final.astype(np.float32),
        "l25_fingertip_positions_optimized": tips_final.astype(np.float32),
        "fingertip_error_before_m": error_before.astype(np.float32),
        "fingertip_error_after_m": error_after.astype(np.float32),
        "collision_optimized_finger_names": np.asarray(sorted(fingers)),
        "collision_optimized_vector_indices": variable_indices.astype(np.int64),
        "collision_penetration_before_m": before_penetrations.astype(np.float32),
        "collision_penetration_after_m": after_penetrations.astype(np.float32),
    })
    np.savez_compressed(args.output, **plan)
    report = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "source_plan": str(args.plan.resolve()),
        "scene_xml": str(args.scene_xml.resolve()),
        "method": "MuJoCo mesh-contact-penalty + active-fingertip-target preservation",
        "optimized_fingers": sorted(fingers),
        "optimized_joint_count": int(len(variable_indices)),
        "max_joint_delta_rad": float(args.max_joint_delta_rad),
        "optimization_success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "cost": float(final_cost),
        "active_fingertip_error_before_mm": (error_before[active] * 1000.0).tolist(),
        "active_fingertip_error_after_mm": (error_after[active] * 1000.0).tolist(),
        "max_penetration_before_mm": float(before_penetrations.max(initial=0.0) * 1000.0),
        "max_penetration_after_mm": float(after_penetrations.max(initial=0.0) * 1000.0),
        "contact_pairs_before": contacts_before,
        "contact_pairs_after": contacts_after,
        "limitations": "The object collision mesh is the simplified MuJoCo proxy. This evaluates hand geometry collision only; it does not establish force closure or real-camera calibration.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Collision-aware L25 object-relative plan written (simulation only)")
    print("  optimized fingers: " + ", ".join(sorted(fingers)))
    print(f"  max penetration [mm]: {report['max_penetration_before_mm']:.3f} -> {report['max_penetration_after_mm']:.3f}")
    print("  active tip error after [mm]: " + ", ".join(f"{value:.3f}" for value in report["active_fingertip_error_after_mm"]))
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
