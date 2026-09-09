#!/usr/bin/env python3
"""Create a hash-bound O30 Vector scale profile from 21-point hand samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from anydexretarget.o30_scale import calibrate_profile, nominal_profile


def _keypoints(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("human_keypoints_canonical", "human_keypoints", "keypoints_canonical", "keypoints"):
            if key in data:
                return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"No 21-point keypoint array found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, help="NPZ containing N x 21 x 3 or 21 x 3 keypoints")
    args = parser.parse_args()
    profile = nominal_profile() if args.keypoints is None else calibrate_profile(_keypoints(args.keypoints))
    profile.save(args.output)
    print(f"O30 scale profile written: {args.output}")
    print(f"  method: {profile.method}; samples: {profile.source_samples}")
    print(f"  vectors: {len(profile.key_vector_scales)}")


if __name__ == "__main__":
    main()
