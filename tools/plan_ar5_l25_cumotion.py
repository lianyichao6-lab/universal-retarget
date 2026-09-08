#!/usr/bin/env python3
"""Generate a non-executing AR5 + L25 grasp trajectory through Luban cuMotion.

The input is a frozen AnyDex grasp request. This tool asks the running Luban
planner for three plan-only segments: pregrasp, final approach, and lift. It
never publishes to an arm or hand controller.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_arm import arm_flange_pose_xyzw, homogeneous_transform
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES, L25_ACTIVE_INDICES


def _read_request(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        request = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {
        "base_frame",
        "T_robot_base_arm_flange_pregrasp",
        "T_robot_base_arm_flange_target",
        "T_robot_base_arm_flange_lift",
        "l25_qpos",
        "l25_preshape_positions",
    }
    missing = required - set(request)
    if missing:
        raise ValueError("grasp request missing: " + ", ".join(sorted(missing)))
    for key in (
        "T_robot_base_arm_flange_pregrasp",
        "T_robot_base_arm_flange_target",
        "T_robot_base_arm_flange_lift",
    ):
        homogeneous_transform(request[key], key)
    if np.asarray(request["l25_qpos"]).shape != (len(L25_QPOS_JOINTS),):
        raise ValueError("l25_qpos must have shape (21,)")
    if np.asarray(request["l25_preshape_positions"]).shape != (16,):
        raise ValueError("l25_preshape_positions must have shape (16,)")
    return request


def _trajectory_from_result(result: Mapping[str, object], start: np.ndarray) -> np.ndarray:
    if not bool(result.get("success")):
        raise RuntimeError(str(result.get("failure_reason") or result.get("status") or "planning failed"))
    raw = result.get("trajectory")
    if not isinstance(raw, Mapping):
        raise RuntimeError("successful cuMotion result did not include trajectory")
    names = tuple(str(name) for name in raw.get("joint_names", ()))
    points = raw.get("points")
    if set(names) != set(AR5_RIGHT_JOINT_NAMES) or not isinstance(points, Sequence) or not points:
        raise RuntimeError("cuMotion returned an invalid AR5 trajectory")
    index = {name: position for position, name in enumerate(names)}
    values = []
    for point in points:
        if not isinstance(point, Mapping):
            raise RuntimeError("cuMotion trajectory point must be an object")
        positions = np.asarray(point.get("positions"), dtype=np.float64)
        if positions.shape != (7,) or not np.isfinite(positions).all():
            raise RuntimeError("cuMotion trajectory point must contain seven finite joints")
        values.append([positions[index[name]] for name in AR5_RIGHT_JOINT_NAMES])
    trajectory = np.asarray(values, dtype=np.float64)
    if not np.allclose(trajectory[0], start, atol=1e-6):
        trajectory = np.vstack((start, trajectory))
    return trajectory


def _append_segment(parts: list[np.ndarray], segment: np.ndarray) -> None:
    if parts and np.allclose(parts[-1][-1], segment[0], atol=1e-6):
        segment = segment[1:]
    if len(segment):
        parts.append(segment)


def _preshape_qpos(active_positions: np.ndarray) -> np.ndarray:
    result = np.zeros(len(L25_QPOS_JOINTS), dtype=np.float32)
    result[L25_ACTIVE_INDICES] = np.asarray(active_positions, dtype=np.float32)
    return result


def _interpolate(first: np.ndarray, second: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        return np.empty((0, len(first)), dtype=np.float32)
    fractions = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
    return (1.0 - fractions) * first + fractions * second


def _hand_timeline(
    phase: np.ndarray,
    final_qpos: np.ndarray,
    preshape_active: np.ndarray,
    close_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if close_frames < 1:
        raise ValueError("close_frames must be positive")
    open_qpos = np.zeros(len(L25_QPOS_JOINTS), dtype=np.float32)
    preshape_qpos = _preshape_qpos(preshape_active)
    close = _interpolate(preshape_qpos, final_qpos, int(np.count_nonzero(phase == "close")))
    close_index = 0
    targets: list[np.ndarray] = []
    for value in phase:
        name = str(value)
        if name == "pregrasp":
            targets.append(open_qpos)
        elif name == "approach":
            targets.append(preshape_qpos)
        elif name == "close":
            targets.append(close[close_index])
            close_index += 1
        elif name == "lift":
            targets.append(final_qpos)
        else:
            raise ValueError(f"unexpected AR5 phase {name!r}")
    return np.asarray(targets, dtype=np.float32), np.asarray(phase)


def _query_segment(
    start: np.ndarray,
    transform: np.ndarray,
    frame_id: str,
    timeout: float,
) -> np.ndarray:
    try:
        from motion_planning_utils import QueryPlanOnlineCumotionResult
    except ImportError as exc:
        raise RuntimeError(
            "source the Luban workspace so motion_planning_utils is importable"
        ) from exc
    position, quaternion = arm_flange_pose_xyzw(transform)
    result = QueryPlanOnlineCumotionResult(
        "right",
        start.tolist(),
        position.tolist(),
        quaternion.tolist(),
        timeout=timeout,
        frame_id=frame_id,
    )
    if result is None:
        raise RuntimeError("cuMotion planner did not answer before timeout")
    return _trajectory_from_result(result, start)


def build_plan(
    request: Mapping[str, np.ndarray],
    start: np.ndarray,
    *,
    timeout: float,
    close_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    if start.shape != (7,) or not np.isfinite(start).all():
        raise ValueError("start joints must contain seven finite values")
    frame_id = str(np.asarray(request["base_frame"]).item())
    current = start
    arm_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    for phase, key in (
        ("pregrasp", "T_robot_base_arm_flange_pregrasp"),
        ("approach", "T_robot_base_arm_flange_target"),
    ):
        segment = _query_segment(current, np.asarray(request[key]), frame_id, timeout)
        before = sum(len(part) for part in arm_parts)
        _append_segment(arm_parts, segment)
        added = sum(len(part) for part in arm_parts) - before
        if added:
            phase_parts.append(np.full(added, phase))
        current = segment[-1]
    arm_parts.append(np.repeat(current[None], close_frames, axis=0))
    phase_parts.append(np.full(close_frames, "close"))
    lift = _query_segment(
        current,
        np.asarray(request["T_robot_base_arm_flange_lift"]),
        frame_id,
        timeout,
    )
    before = sum(len(part) for part in arm_parts)
    _append_segment(arm_parts, lift)
    added = sum(len(part) for part in arm_parts) - before
    if added:
        phase_parts.append(np.full(added, "lift"))
    arm = np.concatenate(arm_parts, axis=0)
    phase = np.concatenate(phase_parts)
    hand, hand_phase = _hand_timeline(
        phase,
        np.asarray(request["l25_qpos"], dtype=np.float32),
        np.asarray(request["l25_preshape_positions"], dtype=np.float32),
        close_frames,
    )
    if arm.shape[0] != hand.shape[0]:
        raise AssertionError("AR5 and L25 timelines must have equal frame counts")
    return arm, phase, hand, hand_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--start-joints", type=float, nargs=7, required=True)
    parser.add_argument("--arm-output", type=Path, required=True)
    parser.add_argument("--l25-output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--close-frames", type=int, default=10)
    args = parser.parse_args()
    if args.fps <= 0.0 or args.timeout <= 0.0:
        parser.error("fps and timeout must be positive")

    request = _read_request(args.request)
    arm, phase, hand, hand_phase = build_plan(
        request,
        np.asarray(args.start_joints, dtype=np.float64),
        timeout=args.timeout,
        close_frames=args.close_frames,
    )
    timestamps = np.arange(len(arm), dtype=np.float64) / args.fps
    args.arm_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.arm_output,
        joint_names=np.asarray(AR5_RIGHT_JOINT_NAMES),
        positions=arm.astype(np.float32),
        timestamps=timestamps,
        phase=phase,
    )
    args.l25_output.parent.mkdir(parents=True, exist_ok=True)
    with args.l25_output.open("wb") as stream:
        pickle.dump(
            [{"target": target, "phase": str(name)} for target, name in zip(hand, hand_phase)],
            stream,
        )
    report = {
        "request": str(args.request.resolve()),
        "frames": int(len(arm)),
        "fps": args.fps,
        "ar5_action_dim": 7,
        "l25_action_dim": 21,
        "phase_counts": {name: int(np.count_nonzero(phase == name)) for name in np.unique(phase)},
        "planning_only": True,
    }
    args.arm_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
