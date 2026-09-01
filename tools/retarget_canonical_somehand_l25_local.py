#!/usr/bin/env python3
"""Retarget a canonical right-hand grasp with SomeHand on the local L25 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anydexretarget.hand_representation import load_canonical_grasp_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/somehand_l25_local_right.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-grasp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from somehand.api import HandFrame, RetargetingEngine

    state = load_canonical_grasp_state(args.canonical_grasp)
    if state.handedness != "right":
        raise ValueError("The local L25 configuration currently supports right-hand canonical grasps only")
    engine = RetargetingEngine.from_config_path(str(CONFIG))
    result = engine.process(HandFrame(state.keypoints_for_retargeting(), None, "right"))
    # Preserve the solver result at float64 so a value exactly at a joint bound
    # does not become a false violation after float32 rounding.
    qpos = np.asarray(result.qpos, dtype=np.float64)
    model = engine.hand_model.model
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    violations = int(np.count_nonzero((qpos < lower - 1e-7) | (qpos > upper + 1e-7)))
    names = np.asarray(engine.hand_model.get_joint_names())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int64),
        backend=np.asarray("somehand_local_l25"),
        robot=np.asarray("l25"),
        human_keypoints=state.keypoints_for_retargeting().astype(np.float32),
        robot_qpos=qpos,
        robot_joint_names=names,
        joint_lower=lower.astype(np.float32),
        joint_upper=upper.astype(np.float32),
    )
    metadata = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "backend": "somehand_local_l25",
        "model": str((ROOT / "assets/linkerhand_l25/linkerhand_l25_right_somehand.xml").resolve()),
        "source_canonical_grasp": str(args.canonical_grasp.resolve()),
        "finite": bool(np.isfinite(qpos).all()),
        "joint_limit_violations": violations,
        "dof": int(qpos.size),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"SomeHand local L25: qpos={qpos.shape}, finite={metadata['finite']}, limit_violations={violations}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
