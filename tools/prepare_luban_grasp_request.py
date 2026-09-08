#!/usr/bin/env python3
"""Freeze a capture-time AnyDex grasp plan into a Luban execution request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anydexretarget.luban_execution import build_luban_grasp_request


def _load_transform(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise ValueError(f"{path} is missing NPZ key {key!r}")
        return np.asarray(data[key], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-contract", type=Path, required=True)
    parser.add_argument("--base-anchor-capture", type=Path, required=True)
    parser.add_argument("--flange-hand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-frame", default="r_base_link")
    parser.add_argument("--anchor-frame", default="hand_camera_color_optical_frame")
    parser.add_argument("--base-anchor-key", default="T_robot_base_anchor_capture")
    parser.add_argument("--flange-hand-key", default="T_arm_flange_l25_hand")
    parser.add_argument("--pregrasp-offset-hand-m", type=float, nargs=3, required=True)
    args = parser.parse_args()
    with np.load(args.grasp_contract, allow_pickle=False) as data:
        contract = {key: np.asarray(data[key]).copy() for key in data.files}
    request = build_luban_grasp_request(
        contract,
        t_robot_base_anchor_capture=_load_transform(
            args.base_anchor_capture, args.base_anchor_key
        ),
        t_arm_flange_l25_hand=_load_transform(args.flange_hand, args.flange_hand_key),
        base_frame=args.base_frame,
        expected_anchor_frame=args.anchor_frame,
        pregrasp_offset_hand_m=args.pregrasp_offset_hand_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **request)
    report = {
        key: (value.item() if value.shape == () else value.tolist())
        for key, value in request.items()
    }
    report["safety"] = "Planning only. Use luban_ros_grasp_execute.py --execute explicitly."
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Luban grasp request written: {args.output}")
    print("  L25 command: 16 active joints in radians")
    print("  arm targets: pregrasp and grasp pose in robot base frame")


if __name__ == "__main__":
    main()
