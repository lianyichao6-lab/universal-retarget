#!/usr/bin/env python3
"""Offline comparison of the four available L25 retargeting backends."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend
from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "vector": ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml",
    "adaptive": ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml",
    **DEX_CONFIGS,
}
OPTIMIZERS = ("vector", "adaptive", "dexpilot", "joint_angle")
MODEL_PATH = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
TIP_OFFSETS = {
    "thumb_distal": np.asarray([-0.008849, -0.000018, 0.030758]),
    "index_distal": np.asarray([-0.015799, -0.000013, 0.022931]),
}


def model_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def run_one(name: str, keypoints: np.ndarray, model: mujoco.MjModel, runs: int) -> dict:
    if name in DEX_CONFIGS:
        backend = DexRetargetBackend(name, hand_side="right")
        solve = lambda: backend.retarget(keypoints)
        source_names = backend.joint_names
        optimizer = backend.optimizer
    else:
        retargeter = Retargeter.from_yaml(str(CONFIGS[name]), hand_side="right")
        solve = lambda: retargeter.retarget_verbose(keypoints, apply_filter=False)
        source_names = retargeter.optimizer.robot.dof_joint_names
        optimizer = retargeter.optimizer

    # Warm up lazy kinematics/model initialization before timing.
    result = solve()
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        result = solve()
        times.append((time.perf_counter() - started) * 1000.0)
    qpos = np.asarray(result[0], dtype=np.float64)
    verbose = result[1]
    by_name = {str(n).lower(): i for i, n in enumerate(source_names)}
    names = model_joint_names(model)
    missing = [n for n in names if n.lower() not in by_name]
    if missing:
        raise ValueError(f"{name} missing MuJoCo joints: {missing}")
    mapped = np.asarray([qpos[by_name[n.lower()]] for n in names])
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    violations = int(np.count_nonzero((mapped < lower) | (mapped > upper)))
    clamped = np.clip(mapped, lower + 1e-6, upper - 1e-6)
    margin = np.minimum((clamped - lower) / (upper - lower), (upper - clamped) / (upper - lower))

    data = mujoco.MjData(model)
    data.qpos[:] = clamped
    mujoco.mj_forward(model, data)
    thumb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb_distal")
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "index_distal")
    if thumb >= 0 and index >= 0:
        thumb_tip = data.xpos[thumb] + data.xmat[thumb].reshape(3, 3) @ TIP_OFFSETS["thumb_distal"]
        index_tip = data.xpos[index] + data.xmat[index].reshape(3, 3) @ TIP_OFFSETS["index_distal"]
        pinch = float(np.linalg.norm(thumb_tip - index_tip))
    else:
        pinch = None
    cost = getattr(getattr(optimizer, "opt", None), "last_optimum_value", lambda: float("nan"))()
    return {
        "robot": "l25",
        "optimizer": name,
        "dof": int(model.njnt),
        "finite_qpos": bool(np.isfinite(qpos).all()),
        "mean_solve_ms": float(np.mean(times)),
        "p95_solve_ms": float(np.percentile(times, 95)),
        "limit_violations_before_clamp": violations,
        "saturated_joint_count_5pct": int(np.count_nonzero(margin <= 0.05)),
        "min_normalized_joint_margin": float(np.min(margin)),
        "thumb_index_distance_m": pinch,
        "solver_cost_same_optimizer_only": float(cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not args.canonical.is_file():
        raise FileNotFoundError(args.canonical)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{args.output}/summary.csv exists; use --overwrite")
    state = load_canonical_grasp_state(args.canonical)
    if state.handedness != "right":
        raise ValueError("L25 benchmark currently supports right hand only")
    keypoints = state.keypoints_for_retargeting()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    rows = [run_one(name, keypoints, model, args.runs) for name in OPTIMIZERS]
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"canonical": str(args.canonical.resolve()), "offline_only": True, "results": rows}, f, indent=2)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} L25 optimizer rows to {args.output}")
    for row in rows:
        pinch = row["thumb_index_distance_m"]
        pinch_text = "N/A" if pinch is None else f"{pinch:.4f} m"
        print(f"  {row['optimizer']:11s} solve={row['mean_solve_ms']:.2f} ms "
              f"limit={row['limit_violations_before_clamp']} sat={row['saturated_joint_count_5pct']} "
              f"pinch={pinch_text}")


if __name__ == "__main__":
    main()
