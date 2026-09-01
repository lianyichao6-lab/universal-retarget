#!/usr/bin/env python3
"""Audit fixed L25/Canonical-MANO bone-length ratios for Adaptive scaling.

This uses consecutive MANO landmark distances, which are pose-invariant bone
lengths, rather than wrist-to-tip distances that change while a hand closes.
It reports four scales per finger: wrist->MCP, MCP->PIP, PIP->DIP and DIP->tip.
No robot configuration is changed unless --config-output is explicitly given.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import yaml

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.l25_target_chain import FINGER_CHAINS, l25_chain_points
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENTS = ("wrist_to_mcp", "mcp_to_pip", "pip_to_dip", "dip_to_tip")


def _reference_q(optimizer) -> np.ndarray:
    q = optimizer.neutral_qpos.copy() if optimizer.neutral_qpos is not None else np.zeros(optimizer.robot.model.nq)
    return np.clip(q, optimizer.robot.joint_limits[:, 0], optimizer.robot.joint_limits[:, 1])


def _source_lengths(state_path: Path, retargeter: Retargeter) -> np.ndarray:
    state = load_canonical_grasp_state(state_path)
    _q, verbose = retargeter.retarget_verbose(state.keypoints_for_retargeting(), apply_filter=False)
    points = np.asarray(verbose["mediapipe_kp"], dtype=np.float64)
    lengths = np.empty((5, 4), dtype=np.float64)
    for finger, indices in enumerate(FINGER_CHAINS):
        lengths[finger] = np.linalg.norm(points[indices[1:]] - points[indices[:-1]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-8):
        raise ValueError(f"Degenerate MANO segment in {state_path}")
    return lengths


def _current_scaling(config: dict) -> np.ndarray:
    result = np.ones((5, 4), dtype=np.float64)
    supplied = config["retarget"].get("segment_scaling", {})
    for index, finger in enumerate(FINGERS):
        values = np.asarray(supplied.get(finger, ()), dtype=np.float64)
        if len(values) == 4:
            result[index] = values
        elif len(values) == 3:
            result[index, 1:] = values
        elif len(values):
            raise ValueError(f"segment_scaling.{finger} must contain three or four values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, action="append", required=True,
                        help="CanonicalGraspState .npz; repeat for a robust median.")
    parser.add_argument("--canonical-glob", action="append", default=[],
                        help="Optional glob(s) of CanonicalGraspState files.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--config-output", type=Path,
                        help="Optional experimental Adaptive YAML with four calibrated values per finger.")
    args = parser.parse_args()
    canonical_paths = list(args.canonical)
    for pattern in args.canonical_glob:
        canonical_paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    canonical_paths = list(dict.fromkeys(canonical_paths))
    if not args.config.is_file() or not canonical_paths or any(not path.is_file() for path in canonical_paths):
        raise FileNotFoundError("--config and every canonical input must exist")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    retargeter = Retargeter.from_yaml(str(args.config), hand_side="right")
    robot_reference = l25_chain_points(retargeter.optimizer, _reference_q(retargeter.optimizer))
    robot_lengths = np.empty((5, 4), dtype=np.float64)
    for finger, indices in enumerate(FINGER_CHAINS):
        robot_lengths[finger] = np.linalg.norm(
            robot_reference[indices[1:]] - robot_reference[indices[:-1]], axis=1
        )

    sample_lengths = np.asarray([_source_lengths(path, retargeter) for path in canonical_paths])
    sample_scales = robot_lengths[None] / sample_lengths
    calibrated = np.median(sample_scales, axis=0)
    low, high = np.percentile(sample_scales, (25, 75), axis=0)
    current = _current_scaling(config)
    rows: list[dict[str, object]] = []
    for finger_index, finger in enumerate(FINGERS):
        for segment_index, segment in enumerate(SEGMENTS):
            rows.append({
                "finger": finger,
                "segment": segment,
                "l25_fk_length_m": float(robot_lengths[finger_index, segment_index]),
                "canonical_median_length_m": float(np.median(sample_lengths[:, finger_index, segment_index])),
                "current_scale": float(current[finger_index, segment_index]),
                "calibrated_scale_median": float(calibrated[finger_index, segment_index]),
                "calibrated_scale_q25": float(low[finger_index, segment_index]),
                "calibrated_scale_q75": float(high[finger_index, segment_index]),
            })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "L25 neutral-FK segment length / Canonical MANO segment length",
        "canonical_sample_count": len(canonical_paths),
        "canonical_inputs": [str(path.resolve()) for path in canonical_paths],
        "config": str(args.config.resolve()),
        "current_segment_scaling_interpretation": "Three values in the source YAML mean [1.0, PIP, DIP, TIP].",
        "recommended_segment_scaling_four_values": {
            finger: calibrated[index].tolist() for index, finger in enumerate(FINGERS)
        },
        "rows": rows,
        "limitations": "This is morphology calibration only. It does not tune contact weights, joint limits, task offsets, or real-hardware zero points.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.report.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.config_output:
        output_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        output_config.setdefault("retarget", {})["segment_scaling"] = {
            finger: [round(float(value), 6) for value in calibrated[index]]
            for index, finger in enumerate(FINGERS)
        }
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")

    print("L25 segment-scaling audit written; source YAML unchanged")
    print(f"  canonical samples: {len(canonical_paths)}")
    print(f"  report: {args.report}")
    if args.config_output:
        print(f"  experimental config: {args.config_output}")
    for index, finger in enumerate(FINGERS):
        values = ", ".join(f"{value:.3f}" for value in calibrated[index])
        print(f"  {finger}: [{values}]")


if __name__ == "__main__":
    main()
