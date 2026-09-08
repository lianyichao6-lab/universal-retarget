#!/usr/bin/env python3
"""Refine an O30 object-relative plan against MuJoCo mesh contacts.

Allowed object contact is limited to distal links. Palm/proximal contact and
cross-finger penetration are penalized while preserving HUG-derived targets.
This is offline simulation only.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _joint_map(model: mujoco.MjModel) -> dict[str, int]:
    return {str(name).lower(): index for index in range(model.njnt) if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))}


def _set_q(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, names: list[str], joint_ids: dict[str, int]) -> None:
    data.qpos[:] = 0.0
    for value, name in zip(qpos, names):
        data.qpos[model.jnt_qposadr[joint_ids[name.lower()]]] = value
    mujoco.mj_forward(model, data)


def _finger(name: str) -> str | None:
    lowered = name.lower()
    return next((finger for finger in FINGERS if lowered.startswith(finger + "_")), None)


def _contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    object_ids = {index for index in range(model.ngeom) if "reconstructed_object" in str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")}
    objects, selves = [], []
    for index in range(data.ncon):
        item = data.contact[index]
        first, second = int(item.geom1), int(item.geom2)
        name1 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first) or f"geom_{first}")
        name2 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second) or f"geom_{second}")
        if first in object_ids or second in object_ids:
            hand = name2 if first in object_ids else name1
            objects.append({"hand_geom": hand, "distance_m": float(item.dist), "position_m": np.asarray(item.pos).tolist()})
            continue
        first_finger, second_finger = _finger(name1), _finger(name2)
        if first_finger and second_finger and first_finger != second_finger:
            selves.append({"geom1": name1, "geom2": name2, "finger1": first_finger, "finger2": second_finger, "distance_m": float(item.dist), "position_m": np.asarray(item.pos).tolist()})
    return objects, selves


def _tip_points(model: mujoco.MjModel, data: mujoco.MjData, links: list[str], offsets: np.ndarray) -> np.ndarray:
    points = []
    for link, offset in zip(links, offsets):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body < 0:
            raise ValueError(f"MuJoCo model lacks O30 fingertip body {link}")
        points.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset)
    return np.asarray(points, dtype=np.float64)


def _penetration(items: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([max(0.0, -float(item["distance_m"])) for item in items], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-weight", type=float, default=22.0)
    parser.add_argument("--self-collision-weight", type=float, default=30.0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--mesh-contact-weight", type=float, default=2.0, help="Weight for actual O30 fingertip-to-displayed-mesh closure.")
    parser.add_argument("--posture-weight", type=float, default=0.08)
    parser.add_argument("--contact-scale-mm", type=float, default=8.0)
    parser.add_argument("--penetration-scale-mm", type=float, default=2.0)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=120)
    args = parser.parse_args()
    if min(args.collision_weight, args.self_collision_weight, args.contact_weight, args.mesh_contact_weight, args.contact_scale_mm, args.penetration_scale_mm, args.max_joint_delta_rad, args.max_iterations) <= 0 or args.posture_weight < 0:
        raise ValueError("Invalid collision refinement settings")
    with np.load(args.plan, allow_pickle=False) as source:
        plan = {key: np.asarray(source[key]).copy() for key in source.files}
    required = {"qpos_vector_order", "vector_joint_names", "active_contact_mask", "contact_target_positions_o30", "o30_fingertip_link_names", "o30_fingertip_task_offsets"}
    if missing := required - set(plan):
        raise ValueError("Plan missing: " + ", ".join(sorted(missing)))
    mesh_path = args.scene_xml.parent / "object_in_o30_simulation_frame.stl"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    object_mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(object_mesh, trimesh.Trimesh):
        raise ValueError("Expected one transformed object triangle mesh")
    model, data = mujoco.MjModel.from_xml_path(str(args.scene_xml)), None
    data = mujoco.MjData(model)
    names = [str(name) for name in plan["vector_joint_names"]]
    q_initial = np.asarray(plan["qpos_vector_order"], dtype=np.float64)
    active = np.asarray(plan["active_contact_mask"], dtype=np.uint8).astype(bool)
    targets = np.asarray(plan["contact_target_positions_o30"], dtype=np.float64)
    links = [str(name) for name in plan["o30_fingertip_link_names"]]
    offsets = np.asarray(plan["o30_fingertip_task_offsets"], dtype=np.float64)
    if q_initial.shape != (len(names),) or targets.shape != (5, 3) or offsets.shape != (5, 3):
        raise ValueError("Invalid O30 plan dimensions")
    joint_ids = _joint_map(model)
    if any(name.lower() not in joint_ids for name in names):
        raise ValueError("Scene and plan joint contracts differ")
    indices = np.asarray([joint_ids[name.lower()] for name in names], dtype=np.int64)
    lower, upper = model.jnt_range[indices, 0], model.jnt_range[indices, 1]
    q_initial = np.clip(q_initial, lower + 1e-6, upper - 1e-6)
    ranges = np.maximum(upper - lower, 1e-6)
    allowed_prefixes = tuple(f"{finger}_distal" for finger in FINGERS)

    def evaluate(qpos: np.ndarray, detailed: bool = False):
        _set_q(model, data, qpos, names, joint_ids)
        points = _tip_points(model, data, links, offsets)
        object_pairs, self_pairs = _contacts(model, data)
        forbidden = [item for item in object_pairs if not str(item["hand_geom"]).lower().startswith(allowed_prefixes)]
        object_pen, self_pen = _penetration(forbidden), _penetration(self_pairs)
        contact_error = (points[active] - targets[active]).reshape(-1) / (args.contact_scale_mm / 1000.0)
        _closest, mesh_distance, _face = trimesh.proximity.closest_point_naive(object_mesh, points)
        mesh_error = mesh_distance[active] / (args.contact_scale_mm / 1000.0)
        posture_error = (qpos - q_initial) / ranges
        cost = (args.contact_weight * float(np.dot(contact_error, contact_error)) + args.mesh_contact_weight * float(np.dot(mesh_error, mesh_error)) + args.posture_weight * float(np.dot(posture_error, posture_error)) + args.collision_weight * float(np.sum((object_pen / (args.penetration_scale_mm / 1000.0)) ** 2)) + args.self_collision_weight * float(np.sum((self_pen / (args.penetration_scale_mm / 1000.0)) ** 2)))
        return (cost, points, object_pairs, self_pairs, forbidden, object_pen, self_pen, mesh_distance) if detailed else cost

    bounds = [(float(max(lower[index] + 1e-6, q_initial[index] - args.max_joint_delta_rad)), float(min(upper[index] - 1e-6, q_initial[index] + args.max_joint_delta_rad))) for index in range(len(q_initial))]
    result = minimize(evaluate, q_initial, method="Powell", bounds=bounds, options={"maxiter": args.max_iterations, "xtol": 1e-4, "ftol": 1e-6})
    before = evaluate(q_initial, detailed=True)
    after = evaluate(np.asarray(result.x, dtype=np.float64), detailed=True)
    q_final, points_before, points_after = np.asarray(result.x, dtype=np.float64), before[1], after[1]
    final_by_name = {name.lower(): value for name, value in zip(names, q_final)}
    plan.update({
        "schema_version": np.asarray(2, dtype=np.int64), "optimizer": np.asarray("o30_object_and_self_collision_aware"),
        "source_o30_object_relative_plan": np.asarray(str(args.plan.resolve())), "qpos_vector_order": q_final.astype(np.float32),
        "qpos": np.asarray([final_by_name[str(name).lower()] for name in plan["robot_joint_names"]], dtype=np.float32),
        "o30_fingertip_positions_collision_refined": points_after.astype(np.float32),
        "contact_error_before_m": np.linalg.norm(points_before - targets, axis=1).astype(np.float32), "contact_error_after_m": np.linalg.norm(points_after - targets, axis=1).astype(np.float32),
        "object_penetration_before_m": before[5].astype(np.float32), "object_penetration_after_m": after[5].astype(np.float32),
        "mesh_fingertip_distance_before_m": before[7].astype(np.float32), "mesh_fingertip_distance_after_m": after[7].astype(np.float32),
        "self_penetration_before_m": before[6].astype(np.float32), "self_penetration_after_m": after[6].astype(np.float32),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **plan)
    trajectory_path = args.output.with_name(args.output.stem + "_trajectory.pkl")
    neutral = np.clip(np.zeros_like(q_final), lower + 1e-6, upper - 1e-6)
    with trajectory_path.open("wb") as stream:
        pickle.dump([{"target": (neutral + fraction * (q_final - neutral)).astype(np.float32), "sim_qpos": (neutral + fraction * (q_final - neutral)).astype(np.float32), "robot_joint_names": names, "robot": "o30", "optimizer": "o30_object_and_self_collision_aware"} for fraction in np.linspace(0.0, 1.0, 30)], stream)
    report = {
        "simulation_only": True, "hardware_command_generated": False, "source_plan": str(args.plan.resolve()), "scene_xml": str(args.scene_xml.resolve()),
        "method": "MuJoCo forbidden-object-penetration + cross-finger-self-collision + O30-tip-target + actual-mesh-closure", "optimization_success": bool(result.success),
        "message": str(result.message), "iterations": int(getattr(result, "nit", 0)), "cost": float(after[0]),
        "active_contact_error_before_mm": (np.linalg.norm(points_before - targets, axis=1)[active] * 1000.0).tolist(),
        "active_contact_error_after_mm": (np.linalg.norm(points_after - targets, axis=1)[active] * 1000.0).tolist(),
        "active_mesh_distance_before_mm": (before[7][active] * 1000.0).tolist(), "active_mesh_distance_after_mm": (after[7][active] * 1000.0).tolist(),
        "forbidden_object_pairs_before": before[4], "forbidden_object_pairs_after": after[4], "self_pairs_before": before[3], "self_pairs_after": after[3],
        "max_forbidden_object_penetration_before_mm": float(before[5].max(initial=0.0) * 1000.0), "max_forbidden_object_penetration_after_mm": float(after[5].max(initial=0.0) * 1000.0),
        "max_self_penetration_before_mm": float(before[6].max(initial=0.0) * 1000.0), "max_self_penetration_after_mm": float(after[6].max(initial=0.0) * 1000.0), "trajectory": str(trajectory_path.resolve()),
        "limitations": "MuJoCo contact reports do not certify force closure, tactile stability, unknown-object geometry, or physical hand calibration.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("O30 object/self-collision-aware plan written (simulation only)")
    print(f"  forbidden object penetration [mm]: {report['max_forbidden_object_penetration_before_mm']:.3f} -> {report['max_forbidden_object_penetration_after_mm']:.3f}")
    print(f"  self penetration [mm]: {report['max_self_penetration_before_mm']:.3f} -> {report['max_self_penetration_after_mm']:.3f}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
