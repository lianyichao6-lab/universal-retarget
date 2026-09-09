#!/usr/bin/env python3
"""Convert a validated O30 grasp bundle to the MuJoCo pickle trajectory format."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from anydexretarget.hand_contract import O30_QPOS_JOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--plan", type=Path, help="Simulation-only rigid plan; cannot be used for hardware")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.bundle is not None:
        with np.load(args.bundle, allow_pickle=False) as data:
            if str(data["schema"].item()) != "anydexretarget.o30_grasp_bundle/v1" or not bool(data["validation_passed"].item()):
                raise ValueError("bundle is not a passed O30 mesh-validation bundle")
            names = tuple(str(item) for item in data["robot_joint_names"])
            qposes = np.asarray(data["qpos_closure"], dtype=np.float64)
        label = "validated_mesh_bundle"
    else:
        with np.load(args.plan, allow_pickle=False) as data:
            names = tuple(str(item) for item in data["vector_joint_names"])
            target = np.asarray(data["qpos_vector_order"], dtype=np.float64)
        qposes = np.stack([fraction * target for fraction in np.linspace(0.0, 1.0, 32)])
        label = "simulation_only_plan"
    if names != O30_QPOS_JOINT_NAMES or qposes.ndim != 2 or qposes.shape[1] != 20:
        raise ValueError("source does not satisfy the audited O30 qpos contract")
    records = [
        {
            "timestamp": index / args.fps,
            "target": qpos.astype(np.float64),
            "robot_joint_names": list(O30_QPOS_JOINT_NAMES),
            "robot": "o30",
            "optimizer": label,
            "dry_run": True,
        }
        for index, qpos in enumerate(qposes)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"MuJoCo O30 trajectory written: {args.output} ({len(records)} frames)")


if __name__ == "__main__":
    main()
