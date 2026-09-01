#!/usr/bin/env python3
"""Evaluate one L25 solution against a continuous MANO-to-L25 target chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.l25_target_chain import (
    build_l25_target_chain,
    chain_error_metrics,
    l25_chain_points,
)
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.canonical.is_file():
        raise FileNotFoundError(args.canonical)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    state = load_canonical_grasp_state(args.canonical)
    if state.handedness != "right":
        raise ValueError("L25 continuous target evaluation is right-hand only")
    retargeter = Retargeter.from_yaml(str(args.config), hand_side="right")
    qpos, verbose = retargeter.retarget_verbose(
        state.keypoints_for_retargeting(), apply_filter=False
    )
    optimizer = retargeter.optimizer
    target = build_l25_target_chain(optimizer, verbose["mediapipe_kp"])
    robot_points = l25_chain_points(optimizer, np.asarray(qpos, dtype=np.float64))
    metrics = chain_error_metrics(target, robot_points)
    result = {
        "canonical": str(args.canonical.resolve()),
        "config": str(args.config.resolve()),
        "representation": "continuous_mano_direction_l25_fk_length_target",
        "optimizer": type(optimizer).__name__,
        "solver_cost": float(verbose["cost"]),
        "metrics": metrics,
        "l25_segment_lengths_m": target.segment_lengths.tolist(),
        "human_source_directions": target.source_directions.tolist(),
        "offline_only": True,
        "note": (
            "These are diagnostics against a continuous geometry target. "
            "The current Vector optimizer does not minimize these metrics directly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"continuous target: point={metrics['mean_point_error_cm']:.3f} cm "
        f"tip={metrics['tip_mean_error_cm']:.3f} cm "
        f"direction={metrics['mean_segment_direction_error_deg']:.2f} deg "
        f"thumb-index={metrics['thumb_index_distance_error_cm']:.3f} cm"
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
