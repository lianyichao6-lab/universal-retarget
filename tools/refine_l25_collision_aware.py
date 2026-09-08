#!/usr/bin/env python3
"""Refine an L25 plan against object and cross-finger collision."""

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
    result = {}
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


def _finger_from_geom(name: str) -> str | None:
    lowered = name.lower()
    for finger in FINGERS:
        if lowered.startswith(finger + "_"):
            return finger
    return None


def _contact_sets(model: mujoco.MjModel, data: mujoco.MjData, object_id: int
                  ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    object_contacts = []
    self_contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
        if object_id in (geom1, geom2):
            hand_name = name2 if geom1 == object_id else name1
            object_contacts.append({
                "hand_geom": str(hand_name),
                "distance_m": float(contact.dist),
                "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
            })
            continue
        finger1 = _finger_from_geom(str(name1))
        finger2 = _finger_from_geom(str(name2))
        if finger1 is None or finger2 is None or finger1 == finger2:
            continue
        self_contacts.append({
            "geom1": str(name1), "geom2": str(name2),
            "finger1": finger1, "finger2": finger2,
            "distance_m": float(contact.dist),
            "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
        })
    return object_contacts, self_contacts


def _collision_fingers(object_contacts: list[dict[str, object]],
                       self_contacts: list[dict[str, object]]) -> set[str]:
    result = set()
    for contact in object_contacts:
        finger = _finger_from_geom(str(contact["hand_geom"]))
        if finger is not None:
            result.add(finger)
    for contact in self_contacts:
        result.add(str(contact["finger1"]))
        result.add(str(contact["finger2"]))
    return result


def _task_points(model: mujoco.MjModel, data: mujoco.MjData,
                 task_names: list[str], offsets: np.ndarray) -> np.ndarray:
    points = []
    for name, offset in zip(task_names, offsets):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model lacks task body: {name}")
        points.append(
            data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset - ROOT_OFFSET
        )
    return np.asarray(points, dtype=np.float64)


def _normalised_margins(q: np.ndarray, lower: np.ndarray,
                        upper: np.ndarray) -> np.ndarray:
    span = np.maximum(upper - lower, 1e-6)
    return np.minimum((q - lower) / span, (upper - q) / span)


def _penetrations(contacts: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [max(0.0, -float(item["distance_m"])) for item in contacts],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vector-config", type=Path, default=VECTOR_CONFIG)
    parser.add_argument("--collision-weight", type=float, default=22.0)
    parser.add_argument("--self-collision-weight", type=float, default=30.0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.08)
    parser.add_argument("--limit-weight", type=float, default=0.04)
    parser.add_argument("--contact-scale-mm", type=float, default=8.0)
    parser.add_argument("--penetration-scale-mm", type=float, default=2.0)
    parser.add_argument("--self-penetration-scale-mm", type=float, default=1.0)
    parser.add_argument("--minimum-margin", type=float, default=0.015)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=160)
    parser.add_argument("--global-search", action="store_true")
    args = parser.parse_args()
    if not args.plan.is_file() or not args.scene_xml.is_file():
        raise FileNotFoundError("--plan and --scene-xml must both exist")
    positive = (
        args.collision_weight, args.self_collision_weight, args.contact_weight,
        args.contact_scale_mm, args.penetration_scale_mm,
        args.self_penetration_scale_mm, args.max_iterations,
        args.max_joint_delta_rad,
    )
    if min(positive) <= 0 or args.posture_weight < 0:
        raise ValueError("Invalid optimization weights or iteration limit")

    plan = _load_plan(args.plan)
    vector_names = [str(value) for value in plan["vector_joint_names"]]
    q_initial = np.asarray(plan["qpos_vector_order"], dtype=np.float64)
    targets = np.asarray(plan["contact_target_positions_l25"], dtype=np.float64)
    active = np.asarray(plan["active_contact_mask"], dtype=np.uint8).astype(bool)
    if q_initial.shape != (len(vector_names),) or targets.shape != (5, 3):
        raise ValueError("Unexpected L25 plan dimensions")

    model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    data = mujoco.MjData(model)
    joint_ids = _joint_map(model)
    vector_joint_ids = np.asarray(
        [joint_ids[name.lower()] for name in vector_names], dtype=np.int64
    )
    lower = model.jnt_range[vector_joint_ids, 0].astype(np.float64)
    upper = model.jnt_range[vector_joint_ids, 1].astype(np.float64)
    q_initial = np.clip(q_initial, lower + 1e-6, upper - 1e-6)
    object_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "reconstructed_object_geom"
    )
    if object_id < 0:
        raise ValueError("Scene has no reconstructed_object_geom")

    retargeter = Retargeter.from_yaml(str(args.vector_config), hand_side="right")
    task_names = [str(value) for value in retargeter.optimizer.task_link_names]
    default_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    contact_offsets = np.asarray(
        plan.get("contact_task_offsets_l25", default_offsets), dtype=np.float64
    )
    if contact_offsets.shape != (5, 3):
        raise ValueError("Expected five L25 contact task offsets")

    _set_q(model, data, q_initial, vector_names, joint_ids)
    object_before, self_before = _contact_sets(model, data, object_id)
    fingers = _collision_fingers(object_before, self_before)
    fingers.update(FINGERS[i] for i, enabled in enumerate(active) if enabled)
    variable_indices = np.asarray([
        i for i, name in enumerate(vector_names)
        if any(name.lower().startswith(finger + "_") for finger in fingers)
    ], dtype=np.int64)
    if len(variable_indices) == 0:
        raise ValueError("Could not identify relevant L25 joints")
    ranges = np.maximum(upper - lower, 1e-6)
    contact_scale = args.contact_scale_mm / 1000.0
    penetration_scale = args.penetration_scale_mm / 1000.0
    self_scale = args.self_penetration_scale_mm / 1000.0

    def evaluate(partial: np.ndarray, detailed: bool = False):
        q = q_initial.copy()
        q[variable_indices] = np.asarray(partial, dtype=np.float64)
        _set_q(model, data, q, vector_names, joint_ids)
        points = _task_points(model, data, task_names, contact_offsets)
        object_pairs, self_pairs = _contact_sets(model, data, object_id)
        object_pen = _penetrations(object_pairs)
        self_pen = _penetrations(self_pairs)
        contact_error = (points[active] - targets[active]).reshape(-1) / contact_scale
        posture_error = (q - q_initial) / ranges
        margin_penalty = np.maximum(
            0.0, args.minimum_margin - _normalised_margins(q, lower, upper)
        )
        cost = (
            args.contact_weight * float(np.dot(contact_error, contact_error))
            + args.posture_weight * float(np.dot(posture_error, posture_error))
            + args.collision_weight * float(np.sum((object_pen / penetration_scale) ** 2))
            + args.self_collision_weight * float(np.sum((self_pen / self_scale) ** 2))
            + args.limit_weight * float(np.dot(margin_penalty, margin_penalty))
        )
        if detailed:
            return cost, q, points, object_pairs, self_pairs, object_pen, self_pen
        return cost

    bounds = [
        (
            float(max(lower[i] + 1e-6, q_initial[i] - args.max_joint_delta_rad)),
            float(min(upper[i] - 1e-6, q_initial[i] + args.max_joint_delta_rad)),
        )
        for i in variable_indices
    ]
    start = q_initial[variable_indices]
    if args.global_search:
        global_result = differential_evolution(
            evaluate, bounds=bounds, seed=7, polish=False,
            maxiter=max(8, args.max_iterations // 4), workers=1,
        )
        start = np.asarray(global_result.x, dtype=np.float64)
    result = minimize(
        evaluate, start, method="Powell", bounds=bounds,
        options={"maxiter": args.max_iterations, "xtol": 1e-4, "ftol": 1e-6},
    )
    _, _, points_before, object_pairs_before, self_pairs_before, object_pen_before, self_pen_before = evaluate(
        q_initial[variable_indices], detailed=True
    )
    final_cost, q_final, points_final, object_pairs_after, self_pairs_after, object_pen_after, self_pen_after = evaluate(
        result.x, detailed=True
    )

    q_model_names = [str(value) for value in plan["robot_joint_names"]]
    vector_by_name = {name.lower(): value for name, value in zip(vector_names, q_final)}
    q_model = np.asarray(
        [vector_by_name[name.lower()] for name in q_model_names], dtype=np.float32
    )
    error_before = np.linalg.norm(points_before - targets, axis=1)
    error_after = np.linalg.norm(points_final - targets, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.update({
        "schema_version": np.asarray(5, dtype=np.int64),
        "optimizer": np.asarray("l25_object_and_self_collision_aware"),
        "source_l25_object_relative_plan": np.asarray(str(args.plan.resolve())),
        "qpos": q_model,
        "qpos_vector_order": q_final.astype(np.float32),
        "l25_contact_positions_optimized": points_final.astype(np.float32),
        "contact_error_before_m": error_before.astype(np.float32),
        "contact_error_after_m": error_after.astype(np.float32),
        "collision_optimized_finger_names": np.asarray(sorted(fingers)),
        "collision_optimized_vector_indices": variable_indices,
        "object_penetration_before_m": object_pen_before.astype(np.float32),
        "object_penetration_after_m": object_pen_after.astype(np.float32),
        "self_penetration_before_m": self_pen_before.astype(np.float32),
        "self_penetration_after_m": self_pen_after.astype(np.float32),
        "collision_penetration_before_m": object_pen_before.astype(np.float32),
        "collision_penetration_after_m": object_pen_after.astype(np.float32),
    })
    np.savez_compressed(args.output, **plan)

    report = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "hardware_reproduction_validated": False,
        "source_plan": str(args.plan.resolve()),
        "scene_xml": str(args.scene_xml.resolve()),
        "method": "object-penetration + cross-finger-self-collision + distal-pad-target",
        "optimized_fingers": sorted(fingers),
        "optimized_joint_count": int(len(variable_indices)),
        "max_joint_delta_rad": float(args.max_joint_delta_rad),
        "optimization_success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "cost": float(final_cost),
        "active_contact_error_before_mm": (error_before[active] * 1000.0).tolist(),
        "active_contact_error_after_mm": (error_after[active] * 1000.0).tolist(),
        "active_fingertip_error_before_mm": (error_before[active] * 1000.0).tolist(),
        "active_fingertip_error_after_mm": (error_after[active] * 1000.0).tolist(),
        "max_penetration_before_mm": float(object_pen_before.max(initial=0.0) * 1000.0),
        "max_penetration_after_mm": float(object_pen_after.max(initial=0.0) * 1000.0),
        "max_self_penetration_before_mm": float(self_pen_before.max(initial=0.0) * 1000.0),
        "max_self_penetration_after_mm": float(self_pen_after.max(initial=0.0) * 1000.0),
        "object_contact_pairs_before": object_pairs_before,
        "object_contact_pairs_after": object_pairs_after,
        "self_contact_pairs_before": self_pairs_before,
        "self_contact_pairs_after": self_pairs_after,
        "contact_pairs_before": object_pairs_before,
        "contact_pairs_after": object_pairs_after,
        "limitations": (
            "MuJoCo contacts are simulation diagnostics and do not establish "
            "force closure, tactile stability, or hardware reproduction."
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    object_before_mm = report["max_penetration_before_mm"]
    object_after_mm = report["max_penetration_after_mm"]
    self_before_mm = report["max_self_penetration_before_mm"]
    self_after_mm = report["max_self_penetration_after_mm"]
    print("Object/self-collision-aware L25 plan written")
    print("  optimized fingers: " + ", ".join(sorted(fingers)))
    print(f"  object penetration [mm]: {object_before_mm:.3f} -> {object_after_mm:.3f}")
    print(f"  self penetration [mm]: {self_before_mm:.3f} -> {self_after_mm:.3f}")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
