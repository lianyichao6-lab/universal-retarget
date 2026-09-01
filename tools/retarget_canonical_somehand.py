#!/usr/bin/env python3
"""Retarget one CanonicalGraspState with somehand (offline only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anydexretarget.hand_representation import load_canonical_grasp_state


ROOT = Path(__file__).resolve().parents[1]
SOMEHAND_ROOT = ROOT / "external" / "somehand"
CONFIGS = {
    "l25": SOMEHAND_ROOT / "configs/retargeting/right/linkerhand_l25_right.yaml",
    "l6": SOMEHAND_ROOT / "configs/retargeting/right/linkerhand_l6_right.yaml",
    "o6": SOMEHAND_ROOT / "configs/retargeting/right/linkerhand_o6_right.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-grasp", type=Path, required=True)
    parser.add_argument("--robot", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=("right",), default="right")
    args = parser.parse_args()

    from somehand.api import HandFrame, RetargetingEngine

    state = load_canonical_grasp_state(args.canonical_grasp)
    if state.handedness != args.hand:
        raise ValueError(
            f"Canonical grasp handedness {state.handedness!r} does not match --hand {args.hand!r}"
        )
    engine = RetargetingEngine.from_config_path(str(CONFIGS[args.robot]))
    result = engine.process(
        HandFrame(state.keypoints_for_retargeting(), None, args.hand)
    )
    qpos = np.asarray(result.qpos, dtype=np.float32)
    model = engine.hand_model.model
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    violations = int(np.count_nonzero((qpos < lower - 1e-7) | (qpos > upper + 1e-7)))
    ranges = np.maximum(upper - lower, 1e-9)
    margins = np.minimum((qpos - lower) / ranges, (upper - qpos) / ranges)
    joint_names = np.asarray(engine.hand_model.get_joint_names())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int64),
        backend=np.asarray("somehand"),
        robot=np.asarray(args.robot),
        handedness=np.asarray(args.hand),
        source_canonical_grasp=np.asarray(str(args.canonical_grasp.resolve())),
        human_keypoints=state.keypoints_for_retargeting().astype(np.float32),
        robot_qpos=qpos,
        robot_joint_names=joint_names,
        joint_lower=lower.astype(np.float32),
        joint_upper=upper.astype(np.float32),
    )
    metadata = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "backend": "somehand",
        "robot": args.robot,
        "dof": int(qpos.size),
        "finite": bool(np.isfinite(qpos).all()),
        "joint_limit_violations": violations,
        "saturated_joints_5pct": int(np.count_nonzero(margins <= 0.05)),
        "config": str(CONFIGS[args.robot].resolve()),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"somehand {args.robot}: qpos={qpos.shape}, finite={metadata['finite']}, "
          f"limit_violations={violations}, saturated={metadata['saturated_joints_5pct']}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
