#!/usr/bin/env python3
"""Run one AR5 + L25 MuJoCo episode from Luban-format arm data.

The arm trajectory is converted to flange poses with the Luban AR5 URDF, then
the existing L25 tactile/lift replay is invoked.  No ROS topic is published.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from anydexretarget.arm_fk import (
    AR5ForwardKinematics,
    DEFAULT_AR5_FLANGE_FRAME,
    reorder_ar5_positions,
)


def _read_arm_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        name_key = next((key for key in ("joint_names", "arm_joint_names") if key in data), None)
        position_key = next((key for key in ("positions", "arm_positions") if key in data), None)
        if name_key is None or position_key is None:
            raise ValueError("arm trajectory requires joint_names/positions or arm_joint_names/arm_positions")
        positions = reorder_ar5_positions(data[name_key], data[position_key])
        timestamps = np.asarray(data["timestamps"], dtype=np.float64) if "timestamps" in data else None
    if timestamps is not None and (timestamps.shape != (len(positions),) or not np.isfinite(timestamps).all()):
        raise ValueError("timestamps must have shape (N,) and contain finite values")
    return positions, timestamps


def _write_fk_output(path: Path, positions: np.ndarray, timestamps: np.ndarray | None, urdf: Path, frame: str) -> None:
    transforms = AR5ForwardKinematics(urdf, frame_name=frame).flange_transforms(positions)
    payload = {
        "T_robot_base_arm_flange": transforms,
        "arm_positions": positions,
        "joint_names": np.asarray([f"r_joint_{index}" for index in range(1, 8)]),
        "frame_id": np.asarray("r_base_link"),
        "flange_frame": np.asarray(frame),
    }
    if timestamps is not None:
        payload["timestamps"] = timestamps
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--l25-trajectory", type=Path, required=True)
    parser.add_argument("--arm-trajectory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flange-frame", default=DEFAULT_AR5_FLANGE_FRAME)
    parser.add_argument("--flange-hand", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    parser.add_argument("--lift-m", type=float, default=0.0)
    parser.add_argument("--kinematic-hold", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.lift_m < 0:
        parser.error("--fps must be positive and --lift-m must be non-negative")

    positions, timestamps = _read_arm_trajectory(args.arm_trajectory)
    with tempfile.TemporaryDirectory(prefix="anydex_ar5_l25_") as temp_dir:
        arm_fk_path = Path(temp_dir) / "arm_flange_trajectory.npz"
        _write_fk_output(arm_fk_path, positions, timestamps, args.urdf, args.flange_frame)
        command = [
            sys.executable,
            str(Path(__file__).with_name("replay_l25_lift_episode.py")),
            "--scene", str(args.scene),
            "--trajectory", str(args.l25_trajectory),
            "--arm-target", str(arm_fk_path),
            "--output", str(args.output),
            "--fps", str(args.fps),
            "--contact-force-threshold", str(args.contact_force_threshold),
        ]
        if args.flange_hand is not None:
            command.extend(["--flange-hand", str(args.flange_hand)])
        if args.lift_m:
            command.extend(["--lift-m", str(args.lift_m)])
        if args.kinematic_hold:
            command.append("--kinematic-hold")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
