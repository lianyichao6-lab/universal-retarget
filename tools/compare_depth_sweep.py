#!/usr/bin/env python3
"""Compare offline L25 trajectories produced with different MediaPipe depth scales."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter
from example.output.sim.mujoco_output import ROBOT_HAND_CONFIGS


def evaluate(path: Path, scale: float) -> dict[str, object]:
    with path.open("rb") as stream:
        trajectory = pickle.load(stream)
    mujoco_qpos = np.stack([frame["target"] for frame in trajectory]).astype(float)
    retargeter = Retargeter.from_yaml(
        PROJECT_ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml",
        hand_side="right",
    )
    pin_names = retargeter.optimizer.robot.dof_joint_names
    mujoco_names = ROBOT_HAND_CONFIGS["linkerhand_l25"]["qpos_joint_names"]
    pin_indices = [mujoco_names.index(name) for name in pin_names]
    qpos = mujoco_qpos[:, pin_indices]
    lower = np.asarray(retargeter.optimizer.opt_lower_bounds)
    upper = np.asarray(retargeter.optimizer.opt_upper_bounds)
    delta = np.abs(np.diff(qpos, axis=0))
    return {
        "depth_scale": scale,
        "frames": int(len(qpos)),
        "dof": int(qpos.shape[1]),
        "finite": bool(np.isfinite(qpos).all()),
        "limit_violations": int(((qpos < lower - 1e-8) | (qpos > upper + 1e-8)).sum()),
        "mean_abs_dq": float(delta.mean()),
        "p95_abs_dq": float(np.percentile(delta, 95)),
        "max_abs_dq": float(delta.max()),
        "trajectory": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/l25/depth_sweep"))
    parser.add_argument("--output", type=Path, default=Path("outputs/l25/depth_sweep.csv"))
    args = parser.parse_args()

    rows = []
    for scale in (0.75, 1.0, 1.25, 1.5):
        tag = str(scale).replace(".", "p")
        path = args.directory / f"l25_vector_depth_{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(evaluate(path, scale))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "depth_scale", "frames", "dof", "finite", "limit_violations",
        "mean_abs_dq", "p95_abs_dq", "max_abs_dq", "trajectory",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("depth_scale,frames,dof,finite,limit_violations,mean_abs_dq,p95_abs_dq,max_abs_dq")
    for row in rows:
        print(
            f"{row['depth_scale']},{row['frames']},{row['dof']},{row['finite']},"
            f"{row['limit_violations']},{row['mean_abs_dq']:.6f},"
            f"{row['p95_abs_dq']:.6f},{row['max_abs_dq']:.6f}"
        )
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
