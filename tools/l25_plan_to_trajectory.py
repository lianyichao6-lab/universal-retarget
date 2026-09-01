#!/usr/bin/env python3
"""Export an object-relative L25 plan's exact qpos as a static trajectory."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    with np.load(args.plan, allow_pickle=False) as data:
        if bool(np.asarray(data["simulation_only"]).item()) is not True:
            raise ValueError("expected a simulation-only L25 plan")
        joint_names = [str(name) for name in data["robot_joint_names"]]
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.shape != (21,) or not np.isfinite(qpos).all():
        raise ValueError(f"expected finite L25 qpos shape (21,), got {qpos.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory = [
        {"frame": index, "target": qpos.copy(), "robot_joint_names": joint_names}
        for index in range(args.frames)
    ]
    with args.output.open("wb") as handle:
        pickle.dump(trajectory, handle)
    print(f"Exported original L25 plan qpos to {args.output} ({args.frames} identical frames)")


if __name__ == "__main__":
    main()
