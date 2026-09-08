#!/usr/bin/env python3
"""Export a calibrated AR5 flange target for the Luban pose planner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anydexretarget.luban_arm import arm_flange_pose_xyzw, arm_flange_target


def _load_transform(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise ValueError(f"{path} is missing NPZ key {key!r}")
            value = data[key]
    return np.asarray(value, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-contract", type=Path, required=True)
    parser.add_argument("--base-anchor", type=Path, required=True)
    parser.add_argument("--flange-hand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-anchor-key", default="T_robot_base_anchor")
    parser.add_argument("--flange-hand-key", default="T_arm_flange_l25_hand")
    args = parser.parse_args()

    with np.load(args.grasp_contract, allow_pickle=False) as data:
        contract = {key: np.asarray(data[key]).copy() for key in data.files}
    target = arm_flange_target(
        contract,
        t_robot_base_anchor=_load_transform(args.base_anchor, args.base_anchor_key),
        t_arm_flange_l25_hand=_load_transform(args.flange_hand, args.flange_hand_key),
    )
    translation, quaternion = arm_flange_pose_xyzw(target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        T_robot_base_arm_flange=target,
        position_xyz=translation,
        orientation_xyzw=quaternion,
        frame_id=np.asarray("r_base_link"),
        child_frame_id=np.asarray("r_flange"),
        source_grasp_contract=np.asarray(str(args.grasp_contract.resolve())),
    )
    print("AR5 flange target written")
    print(f"  position xyz: {np.array2string(translation, precision=6)}")
    print(f"  orientation xyzw: {np.array2string(quaternion, precision=6)}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
