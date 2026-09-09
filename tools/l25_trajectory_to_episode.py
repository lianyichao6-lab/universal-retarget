#!/usr/bin/env python3
"""Convert a staged L25 trajectory pickle to a tactile-free episode NPZ."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS


def load_l25_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    targets = []
    phases = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError(f"trajectory record {index} must contain target")
        target = np.asarray(record["target"], dtype=np.float64)
        if target.shape != (len(L25_QPOS_JOINTS),) or not np.isfinite(target).all():
            raise ValueError(f"trajectory record {index} target must be finite with shape (21,)")
        targets.append(target)
        phases.append(str(record.get("phase", "unspecified")))
    return np.asarray(targets, dtype=np.float32), np.asarray(phases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    qpos, phase = load_l25_trajectory(args.trajectory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=np.arange(len(qpos), dtype=np.float64) / args.fps,
        qpos=qpos,
        phase=phase,
        joint_names=np.asarray(L25_QPOS_JOINTS),
    )
    print(f"Wrote tactile-free L25 episode: {args.output} ({len(qpos)} frames)")


if __name__ == "__main__":
    main()
