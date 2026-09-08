#!/usr/bin/env python3
"""Convert a Luban AR5 joint trajectory NPZ into flange poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anydexretarget.arm_fk import AR5ForwardKinematics, DEFAULT_AR5_FLANGE_FRAME, reorder_ar5_positions


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        name_key = next((key for key in ("joint_names", "arm_joint_names") if key in data), None)
        position_key = next((key for key in ("positions", "arm_positions") if key in data), None)
        if name_key is None or position_key is None:
            raise ValueError("trajectory NPZ requires joint_names/positions or arm_joint_names/arm_positions")
        names = np.asarray(data[name_key]).copy()
        positions = np.asarray(data[position_key], dtype=np.float64).copy()
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).copy() if "timestamps" in data else None
    return names, positions, timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", default=DEFAULT_AR5_FLANGE_FRAME)
    args = parser.parse_args()
    names, raw_positions, timestamps = _read(args.trajectory)
    positions = reorder_ar5_positions(names, raw_positions)
    fk = AR5ForwardKinematics(args.urdf, frame_name=args.frame)
    transforms = fk.flange_transforms(positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "T_robot_base_arm_flange": transforms,
        "arm_positions": positions,
        "joint_names": np.asarray([f"r_joint_{i}" for i in range(1, 8)]),
        "frame_id": np.asarray("r_base_link"),
        "flange_frame": np.asarray(args.frame),
    }
    if timestamps is not None:
        if timestamps.shape != (len(positions),) or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must have shape (N,) and be finite")
        payload["timestamps"] = timestamps
    np.savez_compressed(args.output, **payload)
    print(f"Converted {len(positions)} AR5 points to flange poses")
    print(f"  frame: {args.frame}")
    print(f"  first position: {np.array2string(transforms[0, :3, 3], precision=6)}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
