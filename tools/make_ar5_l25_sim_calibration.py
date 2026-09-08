#!/usr/bin/env python3
"""Place one object-anchored HUG grasp at a reachable AR5 simulation pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from anydexretarget.arm_fk import AR5ForwardKinematics
from anydexretarget.luban_arm import homogeneous_transform


def build_sim_calibration(
    grasp_contract: dict[str, np.ndarray],
    flange_hand: object,
    arm_target: object,
    fk: AR5ForwardKinematics,
) -> dict[str, np.ndarray]:
    """Derive base-to-object from a reachable AR5 flange pose.

    This is explicitly a simulation-scene placement, never an extrinsic
    calibration for hardware use.
    """
    t_anchor_hand = homogeneous_transform(
        grasp_contract["T_anchor_l25_hand"], "T_anchor_l25_hand"
    )
    t_flange_hand = homogeneous_transform(flange_hand, "T_arm_flange_l25_hand")
    target = np.asarray(arm_target, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("arm_target must contain seven finite joints")
    t_base_flange = fk.flange_transform(target)
    t_base_hand = t_base_flange @ t_flange_hand
    t_base_anchor = t_base_hand @ np.linalg.inv(t_anchor_hand)
    position = t_base_anchor[:3, 3]
    rpy = Rotation.from_matrix(t_base_anchor[:3, :3]).as_euler("xyz", degrees=False)
    return {
        "simulation_only": np.asarray(True),
        "hardware_ready": np.asarray(False),
        "T_robot_base_anchor_capture": t_base_anchor,
        "T_robot_base_object": t_base_anchor,
        "T_robot_base_arm_flange_target": t_base_flange,
        "T_robot_base_l25_hand_target": t_base_hand,
        "object_pose_xyz_rpy": np.concatenate((position, rpy)),
        "arm_target_joints": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-contract", type=Path, required=True)
    parser.add_argument("--flange-hand", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--arm-target", type=float, nargs=7, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.grasp_contract, allow_pickle=False) as data:
        contract = {key: np.asarray(data[key]).copy() for key in data.files}
    result = build_sim_calibration(
        contract,
        np.load(args.flange_hand, allow_pickle=False),
        args.arm_target,
        AR5ForwardKinematics(args.urdf),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    report = {key: (value.item() if value.shape == () else value.tolist()) for key, value in result.items()}
    report["safety"] = "Simulation scene placement only; it is not a robot calibration."
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
