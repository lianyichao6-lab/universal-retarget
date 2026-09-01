#!/usr/bin/env python3
"""Diagnose L25 Vector retargeting from one CanonicalGraspState.

This offline report identifies raw joint-limit pressure and the Vector
objectives responsible for the largest red-FK to green-target residuals. It
does not modify YAML files or command hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"
MODEL_PATH = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
# Pinocchio uses float64 while MuJoCo ranges can have float32-level rounding.
LIMIT_TOLERANCE_RAD = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--saturation-threshold", type=float, default=0.05)
    parser.add_argument("--top-joints", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 < args.saturation_threshold < 0.5:
        parser.error("--saturation-threshold must be between 0 and 0.5")
    if args.top_joints <= 0:
        parser.error("--top-joints must be positive")
    return args


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None:
            raise ValueError(f"MuJoCo joint {index} is unnamed")
        names.append(name)
    return names


def _joint_rows(
    raw_qpos: np.ndarray,
    source_names: list[str],
    model: mujoco.MjModel,
    model_names: list[str],
    saturation_threshold: float,
) -> tuple[list[str], int, int]:
    source_by_name = {name.lower(): index for index, name in enumerate(source_names)}
    missing = [name for name in model_names if name.lower() not in source_by_name]
    if missing:
        raise ValueError(f"Retargeter output lacks L25 joints: {missing}")
    lines = [
        "| Joint | Raw qpos (rad) | Lower | Upper | Nearest-limit margin | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    violations = 0
    saturated = 0
    for model_index, name in enumerate(model_names):
        qpos = float(raw_qpos[source_by_name[name.lower()]])
        lower, upper = model.jnt_range[model_index]
        margin = min((qpos - lower) / (upper - lower), (upper - qpos) / (upper - lower))
        if qpos < lower - LIMIT_TOLERANCE_RAD or qpos > upper + LIMIT_TOLERANCE_RAD:
            status = "VIOLATION"
            violations += 1
        elif margin <= saturation_threshold:
            status = "SATURATED"
            saturated += 1
        else:
            status = "OK"
        lines.append(
            f"| `{name}` | {qpos:.5f} | {lower:.5f} | {upper:.5f} | {margin:.2%} | {status} |"
        )
    return lines, violations, saturated


def _vector_rows(
    retargeter: Retargeter,
    raw_qpos: np.ndarray,
    transformed_keypoints: np.ndarray,
    key_vectors: list[dict],
    top_joints: int,
) -> list[str]:
    optimizer = retargeter.optimizer
    required = (
        "_kv_computed_link_indices",
        "_kv_computed_link_offsets",
        "_kv_origin_indices",
        "_kv_task_indices",
        "_compute_target_vectors",
    )
    if not all(hasattr(optimizer, name) for name in required):
        raise TypeError("L25 Vector config did not construct a KeyVectorOptimizer")
    positions_cm = optimizer.robot.compute_points_batch(
        raw_qpos,
        optimizer._kv_computed_link_indices,
        optimizer._kv_computed_link_offsets,
    ) * 100.0
    jacobians_cm = optimizer.robot.compute_all_jacobians_batch_with_offsets(
        raw_qpos,
        optimizer._kv_computed_link_indices,
        optimizer._kv_computed_link_offsets,
    ) * 100.0
    robot_vectors = (
        positions_cm[optimizer._kv_task_indices]
        - positions_cm[optimizer._kv_origin_indices]
    )
    target_vectors = optimizer._compute_target_vectors(transformed_keypoints)
    errors_cm = np.linalg.norm(robot_vectors - target_vectors, axis=1)
    source_names = list(optimizer.robot.dof_joint_names)
    lines = [
        "| # | Robot vector | MP vector | Scale | Weight | Task offset (m) | Error (cm) | Most sensitive joints |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for index, entry in enumerate(key_vectors):
        sensitivity = np.linalg.norm(
            jacobians_cm[optimizer._kv_task_indices[index]]
            - jacobians_cm[optimizer._kv_origin_indices[index]],
            axis=0,
        )
        ranked = np.argsort(-sensitivity)[:top_joints]
        influencing = ", ".join(
            f"`{source_names[joint]}` ({sensitivity[joint]:.2f} cm/rad)"
            for joint in ranked
            if sensitivity[joint] > 1e-7
        ) or "none"
        offset = np.asarray(entry.get("task_offset", [0.0, 0.0, 0.0]), dtype=float)
        lines.append(
            "| {index} | `{origin}` -> `{task}` | {origin_kp} -> {task_kp} | "
            "{scale:.3f} | {weight:.3f} | `{offset}` | {error:.3f} | {influencing} |".format(
                index=index,
                origin=entry["origin"],
                task=entry["task"],
                origin_kp=entry["origin_kp"],
                task_kp=entry["task_kp"],
                scale=float(entry.get("scale", 1.0)),
                weight=float(entry.get("weight", 1.0)),
                offset=_format_vector(offset),
                error=float(errors_cm[index]),
                influencing=influencing,
            )
        )
    return lines


def main() -> None:
    args = _parse_args()
    state = load_canonical_grasp_state(args.canonical)
    if state.handedness != "right":
        raise ValueError("Only right-hand L25 canonical states are supported")
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    key_vectors = config["retarget"]["key_vectors"]
    retargeter = Retargeter.from_config(config, hand_side="right")
    raw_qpos, verbose = retargeter.retarget_verbose(
        state.keypoints_for_retargeting(), apply_filter=False
    )
    raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model_names = _model_joint_names(model)
    joint_lines, violations, saturated = _joint_rows(
        raw_qpos,
        list(retargeter.optimizer.robot.dof_joint_names),
        model,
        model_names,
        args.saturation_threshold,
    )
    vector_lines = _vector_rows(
        retargeter,
        raw_qpos,
        np.asarray(verbose["mediapipe_kp"], dtype=np.float64),
        key_vectors,
        args.top_joints,
    )
    lines = [
        "# L25 Vector Retargeting Diagnosis",
        "",
        f"- Canonical input: `{args.canonical.resolve()}`",
        f"- Config: `{CONFIG_PATH.resolve()}`",
        f"- Solver cost: `{float(verbose['cost']):.6f}`",
        f"- Raw qpos finite: `{bool(np.isfinite(raw_qpos).all())}`",
        f"- Raw joint-limit violations: `{violations}`",
        f"- Saturated joints (margin <= {args.saturation_threshold:.0%}): `{saturated}`",
        "",
        "`VIOLATION` is measured before output clamping with a 1e-6 rad numerical tolerance. `SATURATED` is legal but close "
        "to a physical limit. The task offsets below are current configuration values; "
        "their mesh-derived endpoint evidence is in `reports/l25_tip_offset_audit.md`.",
        "",
        "## Joint Limits",
        "",
        *joint_lines,
        "",
        "## Vector Residuals",
        "",
        "Error is the red L25 FK vector minus the green scaled target vector. Jacobian "
        "sensitivity identifies joints that can change that vector locally; it does not "
        "by itself prove that scale or task offset is wrong.",
        "",
        *vector_lines,
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.report} (raw violations={violations}, saturated={saturated}, "
        f"vectors={len(key_vectors)})"
    )


if __name__ == "__main__":
    main()
