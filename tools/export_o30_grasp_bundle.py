#!/usr/bin/env python3
"""Export the top passed full-mesh O30 candidate as a hardware-gated bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from anydexretarget.hand_contract import O30_QPOS_JOINT_NAMES
from anydexretarget.o30_scale import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    best = ranking.get("best_candidate")
    if not isinstance(best, dict) or not best.get("validation_passed"):
        raise RuntimeError("No full-mesh-validated O30 candidate is available for hardware export")
    candidate = str(best["candidate"])
    trajectory_path = args.candidates_dir / candidate / "trajectory.pkl"
    with trajectory_path.open("rb") as stream:
        records = pickle.load(stream)
    qpos_target = np.asarray(records[-1]["target"], dtype=np.float32)
    if qpos_target.shape != (20,) or tuple(records[-1].get("robot_joint_names", ())) != O30_QPOS_JOINT_NAMES:
        raise ValueError("Validated candidate does not contain the audited O30 qpos contract")
    qpos_open = np.zeros_like(qpos_target)
    closure = np.stack([qpos_open + fraction * (qpos_target - qpos_open) for fraction in np.linspace(0.0, 1.0, 32)]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("anydexretarget.o30_grasp_bundle/v1"),
        validation_passed=np.asarray(True),
        candidate_id=np.asarray(candidate),
        robot_joint_names=np.asarray(O30_QPOS_JOINT_NAMES),
        qpos_target=qpos_target,
        qpos_closure=closure,
        ranking_path=np.asarray(str(args.ranking.resolve())),
        ranking_sha256=np.asarray(file_sha256(args.ranking)),
        object_mesh_sha256=np.asarray(str(ranking["object_mesh_sha256"])),
        scale_profile=np.asarray(str(ranking["scale_profile"])),
    )
    print(f"O30 hardware-gated bundle written: {args.output}")
    print(f"  candidate: {candidate}; closure frames: {len(closure)}")


if __name__ == "__main__":
    main()
