#!/usr/bin/env python3
"""Extract the AR5-to-L25 fixed mount transform from a Luban URDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anydexretarget.urdf_transforms import find_fixed_joint_transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--parent", default="r_tcp")
    parser.add_argument("--child", default="r_hand_base_link")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    transform = find_fixed_joint_transform(args.urdf, parent_link=args.parent, child_link=args.child)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, transform)
    print(f"Extracted {args.parent} -> {args.child}")
    print(np.array2string(transform, precision=8, suppress_small=True))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
