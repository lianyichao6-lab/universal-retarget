#!/usr/bin/env python3
"""Write the Luban L25 URDF's nominal right-hand mount transform.

This transform is for local ``MOCK=1`` validation only. It mirrors the
``r_hand_mount`` joint in Luban's L25 URDF; it is not a substitute for the
measured flange-to-hand calibration required before hardware motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rpy_transform(rpy: object, xyz: object) -> np.ndarray:
    """Return the URDF XYZ/RPY homogeneous transform."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    translation = np.asarray(xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("xyz must contain three finite values")
    if not np.isfinite((roll, pitch, yaw)).all():
        raise ValueError("rpy must contain three finite values")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = translation
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--rpy", type=float, nargs=3, default=(0.0, 0.0, -1.5708))
    args = parser.parse_args()
    transform = rpy_transform(args.rpy, args.xyz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        T_arm_flange_l25_hand=transform,
        source=np.asarray("luban_l25_urdf_nominal_mount"),
        mock_only=np.asarray(True),
    )
    report = {
        "source": "luban_l25_urdf_nominal_mount",
        "mock_only": True,
        "xyz_m": list(map(float, args.xyz)),
        "rpy_rad": list(map(float, args.rpy)),
        "T_arm_flange_l25_hand": transform.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"L25 nominal mock mount written: {args.output}")
    print("  Replace this file with the measured flange-to-L25 transform before hardware use.")


if __name__ == "__main__":
    main()
