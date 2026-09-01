#!/usr/bin/env python3
"""Convert a local L25 SomeHand result into the existing MuJoCo playback format."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


L25_JOINT_NAMES = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    with np.load(args.input, allow_pickle=False) as data:
        qpos = np.asarray(data["robot_qpos"], dtype=np.float64).reshape(-1)
        names = [str(name) for name in data["robot_joint_names"]]
        human_keypoints = np.asarray(data["human_keypoints"], dtype=np.float32)
    by_name = {name: index for index, name in enumerate(names)}
    missing = [name for name in L25_JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"SomeHand result is missing local L25 joints: {missing}")
    target = np.asarray([qpos[by_name[name]] for name in L25_JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("SomeHand qpos contains NaN or Inf")
    records = [
        {
            "target": target.copy(),
            "sim_qpos": target.copy(),
            "human_keypoints": human_keypoints.copy(),
            "human_representation": "canonical_grasp_state",
            "robot": "l25",
            "optimizer": "somehand_local_l25",
        }
        for _ in range(args.frames)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream)
    print(f"saved {len(records)} frames to {args.output} from {args.input}")


if __name__ == "__main__":
    main()
