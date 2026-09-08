#!/usr/bin/env python3
"""Write a fixed world-to-anchor transform for a Luban mock grasp request.

The existing Luban request builder accepts a capture-time anchor transform. In
the CAD-only mock there is no physical camera capture, so this tool writes the
same contract with ``motor_object`` as the anchor frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pose_transform(values: object) -> np.ndarray:
    """Return T_world_anchor for ``x y z roll pitch yaw`` in metres/radians."""
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain six finite values: x y z roll pitch yaw")
    x, y, z, roll, pitch, yaw = pose
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    rotation_y = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rotation_z = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = (x, y, z)
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=float, nargs=6, required=True, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    parser.add_argument("--anchor-frame", default="motor_object")
    parser.add_argument("--base-frame", default="world")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.anchor_frame.strip() or not args.base_frame.strip():
        parser.error("--anchor-frame and --base-frame cannot be empty")
    transform = pose_transform(args.pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        T_robot_base_anchor_capture=transform,
        base_frame=np.asarray(args.base_frame),
        anchor_frame=np.asarray(args.anchor_frame),
        source_type=np.asarray("static_mock_object_pose"),
    )
    report = {
        "base_frame": args.base_frame,
        "anchor_frame": args.anchor_frame,
        "pose_xyz_rpy": list(args.pose),
        "T_robot_base_anchor_capture": transform.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Static anchor pose written: {args.output}")
    print(f"  {args.base_frame} -> {args.anchor_frame}: {args.pose}")


if __name__ == "__main__":
    main()
