#!/usr/bin/env python3
"""Pack one replay output into the fixed AR5(7)+L25(21) action contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_arm(path: Path | None, frames: int) -> np.ndarray:
    if path is None:
        return np.zeros((frames, 7), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        key = "positions" if "positions" in data.files else "arm_positions"
        if key not in data.files:
            raise ValueError("arm trajectory requires positions or arm_positions")
        values = np.asarray(data[key], dtype=np.float32)
    if values.shape != (frames, 7) or not np.isfinite(values).all():
        raise ValueError(f"arm positions must have shape ({frames}, 7) and be finite")
    return values


def pack_episode(tactile_path: Path, output_path: Path, arm_path: Path | None = None) -> dict[str, object]:
    with np.load(tactile_path, allow_pickle=False) as data:
        required = ("timestamps", "qpos", "tactile_contact", "tactile_wrench")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"episode is missing {missing}")
        timestamps = np.asarray(data["timestamps"], dtype=np.float32)
        hand = np.asarray(data["qpos"], dtype=np.float32)
        contact = np.asarray(data["tactile_contact"], dtype=bool)
        wrench = np.asarray(data["tactile_wrench"], dtype=np.float32)
    frames = len(timestamps)
    if timestamps.shape != (frames,) or hand.shape != (frames, 21):
        raise ValueError("timestamps and qpos must have shapes (N,) and (N,21)")
    if contact.shape != (frames, 5) or wrench.shape != (frames, 5, 6):
        raise ValueError("tactile arrays must have shapes (N,5) and (N,5,6)")
    if not np.isfinite(timestamps).all() or not np.isfinite(hand).all() or not np.isfinite(wrench).all():
        raise ValueError("episode arrays must be finite")
    arm = _read_arm(arm_path, frames)
    action = np.concatenate((arm, hand), axis=1)
    observation_tactile = np.concatenate((contact.astype(np.float32), wrench.reshape(frames, -1)), axis=1)
    success = np.full(frames, bool(np.any(contact.sum(axis=1) >= 2)), dtype=bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, timestamps=timestamps, action_28=action, l25_qpos=hand,
                        ar5_qpos=arm, tactile=observation_tactile, tactile_contact=contact,
                        tactile_wrench=wrench, success=success)
    report = {"frames": frames, "action_dim": 28, "tactile_dim": 35,
              "grasp_success": bool(success[-1]), "source": str(tactile_path.resolve())}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactile-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-trajectory", type=Path)
    args = parser.parse_args()
    report = pack_episode(args.tactile_episode, args.output, args.arm_trajectory)
    print(f"Packed {report['frames']} frames; action_dim={report['action_dim']} grasp_success={report['grasp_success']}")


if __name__ == "__main__":
    main()
