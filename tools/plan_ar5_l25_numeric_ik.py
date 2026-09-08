#!/usr/bin/env python3
"""Build a non-executing AR5 + L25 grasp trajectory with bounded AR5 IK."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np

from anydexretarget.arm_ik import AR5NumericalIK
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES
try:  # Support both `python tools/...` and pytest namespace imports.
    from plan_ar5_l25_cumotion import _hand_timeline, _read_request
except ImportError:
    from tools.plan_ar5_l25_cumotion import _hand_timeline, _read_request


def _joint_segment(start: np.ndarray, goal: np.ndarray, max_step_rad: float) -> np.ndarray:
    steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_step_rad)))
    return np.linspace(start, goal, steps + 1, dtype=np.float64)


def build_plan(
    request: Mapping[str, np.ndarray],
    solver: AR5NumericalIK,
    start: np.ndarray,
    *,
    close_frames: int,
    max_joint_step_rad: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float | int]]]:
    if max_joint_step_rad <= 0.0 or max_nfev < 1:
        raise ValueError("max_joint_step_rad must be positive and max_nfev must be at least one")
    current = np.asarray(start, dtype=np.float64)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("start joints must contain seven finite values")
    arm_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    diagnostics: dict[str, dict[str, float | int]] = {}
    for phase, key in (
        ("pregrasp", "T_robot_base_arm_flange_pregrasp"),
        ("approach", "T_robot_base_arm_flange_target"),
    ):
        result = solver.solve(request[key], current, max_nfev=max_nfev)
        segment = _joint_segment(current, result.positions, max_joint_step_rad)
        if arm_parts:
            segment = segment[1:]
        arm_parts.append(segment)
        phase_parts.append(np.full(len(segment), phase))
        diagnostics[phase] = {
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "iterations": result.iterations,
        }
        current = result.positions
    arm_parts.append(np.repeat(current[None], close_frames, axis=0))
    phase_parts.append(np.full(close_frames, "close"))
    result = solver.solve(request["T_robot_base_arm_flange_lift"], current, max_nfev=max_nfev)
    segment = _joint_segment(current, result.positions, max_joint_step_rad)[1:]
    arm_parts.append(segment)
    phase_parts.append(np.full(len(segment), "lift"))
    diagnostics["lift"] = {
        "position_error_m": result.position_error_m,
        "orientation_error_rad": result.orientation_error_rad,
        "iterations": result.iterations,
    }
    arm = np.concatenate(arm_parts, axis=0)
    phase = np.concatenate(phase_parts)
    hand, _ = _hand_timeline(
        phase,
        np.asarray(request["l25_qpos"], dtype=np.float32),
        np.asarray(request["l25_preshape_positions"], dtype=np.float32),
        close_frames,
    )
    if len(arm) != len(hand):
        raise AssertionError("AR5 and L25 timelines must have equal frame counts")
    return arm, phase, hand, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--start-joints", type=float, nargs=7, required=True)
    parser.add_argument("--arm-output", type=Path, required=True)
    parser.add_argument("--l25-output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--close-frames", type=int, default=10)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--max-nfev", type=int, default=300, help="Maximum evaluations for each bounded IK solve")
    args = parser.parse_args()
    if args.fps <= 0.0 or args.close_frames < 1:
        parser.error("fps and close-frames must be positive")
    request = _read_request(args.request)
    arm, phase, hand, diagnostics = build_plan(
        request,
        AR5NumericalIK(args.urdf),
        np.asarray(args.start_joints, dtype=np.float64),
        close_frames=args.close_frames,
        max_joint_step_rad=args.max_joint_step_rad,
        max_nfev=args.max_nfev,
    )
    timestamps = np.arange(len(arm), dtype=np.float64) / args.fps
    args.arm_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.arm_output,
        joint_names=np.asarray(AR5_RIGHT_JOINT_NAMES),
        positions=arm.astype(np.float32),
        timestamps=timestamps,
        phase=phase,
        planner=np.asarray("bounded_numerical_ik"),
    )
    args.l25_output.parent.mkdir(parents=True, exist_ok=True)
    with args.l25_output.open("wb") as stream:
        pickle.dump(
            [{"target": target, "phase": str(name)} for target, name in zip(hand, phase)],
            stream,
        )
    report = {
        "request": str(args.request.resolve()),
        "urdf": str(args.urdf.resolve()),
        "frames": int(len(arm)),
        "fps": args.fps,
        "ar5_action_dim": 7,
        "l25_action_dim": 21,
        "planner": "bounded_numerical_ik",
        "max_nfev": args.max_nfev,
        "planning_only": True,
        "phase_counts": {name: int(np.count_nonzero(phase == name)) for name in np.unique(phase)},
        "ik": diagnostics,
    }
    args.arm_output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
