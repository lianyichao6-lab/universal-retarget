#!/usr/bin/env python3
"""Compare canonical HUG grasps with L25 Vector and Adaptive retargeting.

This is an offline diagnostic. It never connects to or commands hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "vector": ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml",
    "adaptive": ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml",
}
MODEL_PATH = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
# Pinocchio uses float64 while the MuJoCo XML range is loaded through float32.
# Treat sub-microradian boundary differences as the same physical limit.
LIMIT_TOLERANCE_RAD = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument(
        "--vector-config",
        type=Path,
        default=CONFIGS["vector"],
        help="Vector YAML to evaluate; defaults to the production L25 configuration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=3,
        help="Number of reset, post-warm-up solves used for timing statistics.",
    )
    args = parser.parse_args()
    if args.timing_runs <= 0:
        parser.error("--timing-runs must be positive")
    return args


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None:
            raise ValueError(f"L25 MuJoCo joint {index} has no name")
        names.append(name)
    return names


def _vector_error_cm(
    retargeter: Retargeter,
    qpos: np.ndarray,
    transformed_keypoints: np.ndarray,
) -> tuple[float | None, float | None]:
    """Return red-FK to green-target vector error for Vector only."""
    optimizer = retargeter.optimizer
    required = (
        "_kv_computed_link_indices",
        "_kv_computed_link_offsets",
        "_kv_origin_indices",
        "_kv_task_indices",
        "_compute_target_vectors",
    )
    if not all(hasattr(optimizer, name) for name in required):
        return None, None

    positions_cm = optimizer.robot.compute_points_batch(
        qpos,
        optimizer._kv_computed_link_indices,
        optimizer._kv_computed_link_offsets,
    ) * 100.0
    robot_vectors = (
        positions_cm[optimizer._kv_task_indices]
        - positions_cm[optimizer._kv_origin_indices]
    )
    target_vectors = optimizer._compute_target_vectors(transformed_keypoints)
    errors_cm = np.linalg.norm(robot_vectors - target_vectors, axis=1)
    return float(np.mean(errors_cm)), float(np.percentile(errors_cm, 95))


def _evaluate_one(
    label: str,
    state_path: Path,
    optimizer_name: str,
    config_path: Path,
    model: mujoco.MjModel,
    model_joint_names: list[str],
    timing_runs: int,
) -> dict[str, object]:
    state = load_canonical_grasp_state(state_path)
    if state.handedness != "right":
        raise ValueError(f"{state_path} is {state.handedness!r}; L25 is right-hand only")
    keypoints = state.keypoints_for_retargeting()
    retargeter = Retargeter.from_yaml(str(config_path), hand_side="right")

    # Exclude lazy first-call setup. Every measured call starts from the same
    # state, so temporal regularization and filtering cannot skew a comparison.
    retargeter.retarget_verbose(keypoints, apply_filter=False)
    retargeter.reset()
    solve_times_ms = []
    for _ in range(timing_runs):
        retargeter.reset()
        started = time.perf_counter()
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        solve_times_ms.append((time.perf_counter() - started) * 1000.0)
    qpos = np.asarray(qpos, dtype=np.float64)
    if not np.isfinite(qpos).all():
        raise ValueError(f"{label}/{optimizer_name} produced NaN or Inf qpos")

    source_names = [str(name).lower() for name in retargeter.optimizer.robot.dof_joint_names]
    source_by_name = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in model_joint_names if name.lower() not in source_by_name]
    if missing:
        raise ValueError(f"Retargeter is missing MuJoCo L25 joints: {missing}")
    model_qpos = np.asarray(
        [qpos[source_by_name[name.lower()]] for name in model_joint_names],
        dtype=np.float64,
    )
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    violations_before = int(
        np.count_nonzero(
            (model_qpos < lower - LIMIT_TOLERANCE_RAD)
            | (model_qpos > upper + LIMIT_TOLERANCE_RAD)
        )
    )
    clamped_qpos = np.clip(model_qpos, lower + 1e-6, upper - 1e-6)
    violations_after = int(
        np.count_nonzero((clamped_qpos < lower) | (clamped_qpos > upper))
    )
    normalized_margin = np.minimum(
        (clamped_qpos - lower) / np.maximum(upper - lower, 1e-9),
        (upper - clamped_qpos) / np.maximum(upper - lower, 1e-9),
    )

    optimizer = retargeter.optimizer
    tip_indices = [optimizer.robot.get_link_index(name) for name in optimizer.task_link_names]
    fingertips = optimizer.robot.compute_points_batch(
        qpos, tip_indices, optimizer.task_offsets
    )
    if fingertips.shape[0] < 2:
        raise ValueError("L25 retargeter did not expose thumb and index task points")
    vector_mean_cm, vector_p95_cm = _vector_error_cm(
        retargeter, qpos, np.asarray(verbose["mediapipe_kp"], dtype=np.float64)
    )
    return {
        "input": label,
        "canonical_grasp": str(state_path.resolve()),
        "optimizer": optimizer_name,
        "optimizer_config": str(config_path.resolve()),
        "dof": int(len(model_joint_names)),
        "finite_qpos": True,
        "mean_solve_ms": float(np.mean(solve_times_ms)),
        "p95_solve_ms": float(np.percentile(solve_times_ms, 95)),
        "max_solve_ms": float(np.max(solve_times_ms)),
        "solver_cost": float(verbose["cost"]),
        "limit_violations_before_clamp": violations_before,
        "limit_violations_after_clamp": violations_after,
        "saturated_joint_count": int(np.count_nonzero(normalized_margin <= 0.05)),
        "min_normalized_joint_margin": float(np.min(normalized_margin)),
        "thumb_index_distance_m": float(np.linalg.norm(fingertips[0] - fingertips[1])),
        "vector_target_mean_error_cm": vector_mean_cm,
        "vector_target_p95_error_cm": vector_p95_cm,
    }


def _write_results(output: Path, rows: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "robot": "l25",
                "offline_only": True,
                "notes": {
                    "solver_cost": "Comparable only within the same optimizer.",
                    "vector_target_error": (
                        "Vector only: red L25 FK task vector versus green scaled target. "
                        "Adaptive uses a different objective, so this field is null."
                    ),
                },
                "results": rows,
            },
            stream,
            indent=2,
        )
    fieldnames = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if not args.canonical.is_file():
        raise FileNotFoundError(args.canonical)
    if not args.vector_config.is_file():
        raise FileNotFoundError(args.vector_config)
    if args.adapted is not None and not args.adapted.is_file():
        raise FileNotFoundError(args.adapted)
    summary = args.output / "summary.json"
    if summary.exists() and not args.overwrite:
        raise FileExistsError(f"{summary} exists; pass --overwrite to replace it")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model_joint_names = _model_joint_names(model)
    inputs = [("canonical", args.canonical)]
    if args.adapted is not None:
        inputs.append(("l25_adapted", args.adapted))
    config_paths = {"vector": args.vector_config, "adaptive": CONFIGS["adaptive"]}
    rows = [
        _evaluate_one(
            label,
            state_path,
            optimizer_name,
            config_paths[optimizer_name],
            model,
            model_joint_names,
            args.timing_runs
        )
        for label, state_path in inputs
        for optimizer_name in ("vector", "adaptive")
    ]
    _write_results(args.output, rows)
    print(f"Wrote {len(rows)} offline L25 evaluation rows to {args.output}")
    for row in rows:
        vector_error = row["vector_target_mean_error_cm"]
        vector_text = "N/A" if vector_error is None else f"{vector_error:.3f} cm"
        print(
            f"  {row['input']:12s} {row['optimizer']:8s} "
            f"cost={row['solver_cost']:.4f} solve={row['mean_solve_ms']:.2f} ms "
            f"sat={row['saturated_joint_count']} pre_limit={row['limit_violations_before_clamp']} "
            f"thumb-index={row['thumb_index_distance_m']:.4f} m vector_error={vector_text}"
        )


if __name__ == "__main__":
    main()
