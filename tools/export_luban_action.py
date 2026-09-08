#!/usr/bin/env python3
"""Export one validated AR5/L25 action for a Luban ROS2 executor.

This command is intentionally hardware-free. It validates the action contract
and writes the 7-DoF AR5 target plus the 16 active L25 commands, which can be
consumed by a ROS2 sender after the plan has passed operator review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anydexretarget.luban_action import action_contract, build_luban_action


def _load_vector(path: Path, key: str, size: int) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{path} does not contain '{key}'")
            values = np.asarray(data[key], dtype=np.float64)
    else:
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain finite shape ({size},), got {values.shape}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l25-qpos", type=Path, required=True,
                        help="NPY vector or NPZ containing qpos (21 internal joints)")
    parser.add_argument("--arm-qpos", type=Path, required=True,
                        help="NPY vector or NPZ containing arm_qpos (7 joints)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-from-start", type=float, default=1.0)
    args = parser.parse_args()

    l25_qpos = _load_vector(args.l25_qpos, "qpos", 21)
    arm_qpos = _load_vector(args.arm_qpos, "arm_qpos", 7)
    action = build_luban_action(
        arm_qpos, l25_qpos, time_from_start_s=args.time_from_start
    )
    result = {
        "schema_version": 1,
        "planning_only": True,
        "arm_positions": action.arm_positions.astype(np.float32),
        "hand_positions": action.hand_positions.astype(np.float32),
        "time_from_start_s": np.asarray(action.time_from_start_s),
        "arm_joint_names": np.asarray(action_contract()["arm_joint_names"]),
        "hand_joint_names": np.asarray(action_contract()["hand_joint_names"]),
        "arm_action": np.asarray(action_contract()["arm_action"]),
        "hand_controller": np.asarray(action_contract()["hand_controller"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    report = {}
    for key, value in result.items():
        array = np.asarray(value)
        report[key] = array.item() if array.shape == () else array.tolist()
    report["safety"] = "planning-only export; no ROS publisher or hardware command"
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Luban action exported: {args.output}")
    print("  AR5 joints: 7")
    print("  L25 active joints: 16 (from internal 21-qpos)")
    print("  planning-only: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
