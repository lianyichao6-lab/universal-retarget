#!/usr/bin/env python3
"""Export one passed rigid O30 plan to the HOP hardware bundle contract."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation.get("schema") not in {"anydexretarget.o30_mesh_plan_validation/v1", "anydexretarget.o30_mesh_plan_validation/v2"} or not validation.get("validation_passed"):
        raise RuntimeError("O30 plan has not passed strict full-mesh validation")
    with np.load(args.plan, allow_pickle=False) as data:
        plan = {key: np.asarray(data[key]).copy() for key in data.files}
    if tuple(str(item) for item in plan["vector_joint_names"]) != O30_QPOS_JOINT_NAMES:
        raise ValueError("O30 plan vector joint order is incompatible with HOP export")
    target = np.asarray(plan["qpos_vector_order"], dtype=np.float32)
    closure = np.stack([fraction * target for fraction in np.linspace(0.0, 1.0, 32)]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("anydexretarget.o30_grasp_bundle/v1"),
        validation_passed=np.asarray(True),
        candidate_id=np.asarray("rigid_plan"),
        robot_joint_names=np.asarray(O30_QPOS_JOINT_NAMES),
        qpos_target=target,
        qpos_closure=closure,
        plan=np.asarray(str(args.plan.resolve())),
        plan_sha256=np.asarray(file_sha256(args.plan)),
        validation=np.asarray(str(args.validation.resolve())),
        validation_sha256=np.asarray(file_sha256(args.validation)),
        object_mesh_sha256=np.asarray(str(validation["object_mesh_sha256"])),
        scale_profile=plan.get("scale_profile", np.asarray("nominal_yaml_baseline")),
    )
    print(f"O30 hardware-gated rigid-plan bundle written: {args.output}")


if __name__ == "__main__":
    main()
