#!/usr/bin/env python3
"""Export a final L25 plan as a calibration-ready arm/hand execution contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from anydexretarget.deployment import build_grasp_execution_plan


def _strict_json_value(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    return value


def _json_value(value: np.ndarray) -> object:
    array = np.asarray(value)
    raw = array.item() if array.shape == () else array.tolist()
    return _strict_json_value(raw)


def _relative_reference(path: Path | None, output: Path) -> Path | None:
    if path is None:
        return None
    return Path(os.path.relpath(path.resolve(), output.parent.resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument("--reconstruction-result", type=Path)
    parser.add_argument("--anchor-frame", default="waist_camera_color_optical_frame")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--candidate-id")
    parser.add_argument(
        "--pregrasp-offset-hand-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional local L25-hand translation; omitted until target-machine validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.plan, allow_pickle=False) as data:
        final_plan = {key: np.asarray(data[key]).copy() for key in data.files}
    candidate_id = args.candidate_id or args.plan.parent.name
    result = build_grasp_execution_plan(
        final_plan,
        source_plan=args.plan,
        anchor_frame=args.anchor_frame,
        hand_side=args.hand_side,
        candidate_id=candidate_id,
        object_mesh=_relative_reference(args.object_mesh, args.output),
        reconstruction_result=_relative_reference(args.reconstruction_result, args.output),
        pregrasp_offset_hand_m=(
            None
            if args.pregrasp_offset_hand_m is None
            else np.asarray(args.pregrasp_offset_hand_m, dtype=np.float64)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    report = {key: _json_value(value) for key, value in result.items()}
    report["limitations"] = (
        "Planning only. Target-machine camera extrinsics, flange-to-hand calibration, "
        "arm IK, collision checking, operator confirmation and emergency stop remain required."
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Grasp execution contract written")
    print(f"  candidate: {candidate_id}")
    print(f"  anchor frame: {args.anchor_frame}")
    print(f"  pregrasp defined: {bool(result['pregrasp_defined'].item())}")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
