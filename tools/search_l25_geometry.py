#!/usr/bin/env python3
"""Search conservative thumb/index scale multipliers for one L25 canonical grasp.

This is an offline experiment. It keeps each digit's internal scale ratios
intact and never changes the production YAML or sends hardware commands.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"
MODEL_PATH = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
TIP_INDICES = (4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config-output",
        type=Path,
        help="Optional experimental YAML written with the best scale multipliers.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--thumb-multipliers",
        type=float,
        nargs="+",
        default=(0.85, 0.925, 1.0, 1.075, 1.15),
    )
    parser.add_argument(
        "--index-multipliers",
        type=float,
        nargs="+",
        default=(0.85, 0.925, 1.0, 1.075, 1.15),
    )
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.thumb_multipliers + args.index_multipliers):
        parser.error("all scale multipliers must be positive")
    return args


def vector_error_cm(retargeter: Retargeter, qpos: np.ndarray, keypoints: np.ndarray) -> float:
    optimizer = retargeter.optimizer
    points = optimizer.robot.compute_points_batch(
        qpos, optimizer._kv_computed_link_indices, optimizer._kv_computed_link_offsets
    )
    robot = points[optimizer._kv_task_indices] - points[optimizer._kv_origin_indices]
    target = optimizer._compute_target_vectors(keypoints) / 100.0
    return float(np.mean(np.linalg.norm(robot - target, axis=1)) * 100.0)


def evaluate(
    base_config: dict, keypoints: np.ndarray, human_pinch_m: float, model: mujoco.MjModel,
    thumb_multiplier: float, index_multiplier: float,
) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    vectors = config["retarget"]["key_vectors"]
    for vector in vectors[:3]:
        vector["scale"] = float(vector.get("scale", 1.0)) * thumb_multiplier
    for vector in vectors[3:6]:
        vector["scale"] = float(vector.get("scale", 1.0)) * index_multiplier

    retargeter = Retargeter.from_config(config, hand_side="right")
    qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
    qpos = np.asarray(qpos, dtype=np.float64)
    optimizer = retargeter.optimizer
    tip_links = [optimizer.robot.get_link_index(name) for name in optimizer.task_link_names]
    tips = optimizer.robot.compute_points_batch(qpos, tip_links, optimizer.task_offsets)
    robot_pinch_m = float(np.linalg.norm(tips[0] - tips[1]))

    source = {name.lower(): index for index, name in enumerate(optimizer.robot.dof_joint_names)}
    model_qpos = np.asarray(
        [qpos[source[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).lower()]] for i in range(model.njnt)],
        dtype=np.float64,
    )
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    margin = np.minimum((model_qpos - lower) / (upper - lower), (upper - model_qpos) / (upper - lower))
    violations = int(np.count_nonzero((model_qpos < lower - 1e-6) | (model_qpos > upper + 1e-6)))
    saturated = int(np.count_nonzero(margin <= 0.05))
    residual_cm = vector_error_cm(retargeter, qpos, np.asarray(verbose["mediapipe_kp"], dtype=np.float64))
    pinch_error_cm = abs(robot_pinch_m - human_pinch_m) * 100.0
    # Ranking is an explicit trade-off, not a physical grasp-success score.
    score = residual_cm + 0.5 * pinch_error_cm + 0.25 * saturated + 2.0 * violations
    return {
        "thumb_multiplier": thumb_multiplier,
        "index_multiplier": index_multiplier,
        "solver_cost": float(verbose["cost"]),
        "vector_mean_error_cm": residual_cm,
        "human_thumb_index_distance_cm": human_pinch_m * 100.0,
        "robot_thumb_index_distance_cm": robot_pinch_m * 100.0,
        "thumb_index_error_cm": pinch_error_cm,
        "limit_violations_before_clamp": violations,
        "saturated_joint_count": saturated,
        "rank_score": score,
    }


def main() -> None:
    args = parse_args()
    state = load_canonical_grasp_state(args.canonical)
    if state.handedness != "right":
        raise ValueError("L25 geometry search is currently right-hand only")
    keypoints = state.keypoints_for_retargeting()
    human_pinch_m = float(np.linalg.norm(keypoints[TIP_INDICES[0]] - keypoints[TIP_INDICES[1]]))
    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    rows = [
        evaluate(base_config, keypoints, human_pinch_m, model, thumb, index)
        for thumb in args.thumb_multipliers
        for index in args.index_multipliers
    ]
    rows.sort(key=lambda row: row["rank_score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best = rows[0]
    if args.config_output is not None:
        best_config = copy.deepcopy(base_config)
        for vector in best_config["retarget"]["key_vectors"][:3]:
            vector["scale"] = round(
                float(vector.get("scale", 1.0)) * float(best["thumb_multiplier"]), 6
            )
        for vector in best_config["retarget"]["key_vectors"][3:6]:
            vector["scale"] = round(
                float(vector.get("scale", 1.0)) * float(best["index_multiplier"]), 6
            )
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(
            yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "canonical": str(args.canonical.resolve()),
                "base_config": str(args.config.resolve()),
                "objective": (
                    "vector_mean_error_cm + 0.5 * thumb_index_error_cm + "
                    "0.25 * saturated_joint_count + 2.0 * limit_violations"
                ),
                "offline_only": True,
                "results": rows,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "best: "
        f"thumb={best['thumb_multiplier']:.3f} index={best['index_multiplier']:.3f} "
        f"vector_error={best['vector_mean_error_cm']:.3f} cm "
        f"pinch_error={best['thumb_index_error_cm']:.3f} cm "
        f"saturated={best['saturated_joint_count']} score={best['rank_score']:.3f}"
    )
    print(f"results: {args.output.with_suffix('.csv')}")
    if args.config_output is not None:
        print(f"experimental config: {args.config_output}")


if __name__ == "__main__":
    main()
