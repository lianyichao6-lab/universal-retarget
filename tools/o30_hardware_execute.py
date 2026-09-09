#!/usr/bin/env python3
"""Safely execute a calibrated O30 trajectory or validated grasp bundle over HOP."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from anydexretarget.hand_contract import O30_ACTIVE_JOINT_NAMES, O30_QPOS_JOINT_NAMES, o30_qpos_to_command_order
from anydexretarget.o30_hardware_profile import O30HardwareProfile


def _driver(path: Path):
    spec = importlib.util.spec_from_file_location("o30_hop", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HOP driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(path: Path, all_frames: bool, frame: int) -> list[np.ndarray]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("O30 trajectory must be a non-empty pickle list")
    selected = records if all_frames else [records[frame]]
    targets = []
    for record in selected:
        qpos = np.asarray(record.get("target"), dtype=np.float64)
        names = tuple(record.get("robot_joint_names", ()))
        if qpos.shape != (20,) or names != O30_QPOS_JOINT_NAMES or not np.isfinite(qpos).all():
            raise ValueError("Trajectory does not satisfy the audited O30 qpos contract")
        targets.append(qpos)
    return targets


def _bundle(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != "anydexretarget.o30_grasp_bundle/v1" or not bool(data["validation_passed"].item()):
            raise ValueError("Bundle is not a passed O30 full-mesh validation bundle")
        if tuple(str(item) for item in data["robot_joint_names"]) != O30_QPOS_JOINT_NAMES:
            raise ValueError("Bundle joint order is not the audited O30 contract")
        frames = np.asarray(data["qpos_closure"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 20 or not np.isfinite(frames).all():
        raise ValueError("Bundle closure trajectory is invalid")
    return [frame for frame in frames]


def _bounded_commands(start: np.ndarray, target: np.ndarray, max_step: int) -> list[np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != (20,) or target.shape != (20,) or max_step <= 0:
        raise ValueError("Invalid O30 bounded command inputs")
    count = max(1, int(np.ceil(np.max(np.abs(target - start)) / max_step)))
    return [np.rint(start + fraction * (target - start)).astype(np.uint8) for fraction in np.linspace(1.0 / count, 1.0, count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--calibration-profile", type=Path, required=True)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-step-counts", type=int, default=2)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--read-state", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_step_counts <= 0:
        parser.error("--fps and --max-step-counts must be positive")
    if args.execute and args.confirm != "O30_HAND_CLEAR":
        parser.error("--execute requires --confirm O30_HAND_CLEAR")
    if (args.execute or args.read_state) and args.driver is None:
        parser.error("--driver is required for --read-state or --execute")
    profile = O30HardwareProfile.load(args.calibration_profile)
    qposes = _bundle(args.bundle) if args.bundle is not None else _trajectory(args.trajectory, args.all_frames, args.frame)
    commands = [profile.command_from_qpos(o30_qpos_to_command_order(qpos)) for qpos in qposes]
    print({"planning_only": not args.execute, "frames": len(commands), "joint_names": list(O30_ACTIVE_JOINT_NAMES), "first_command": commands[0].tolist(), "last_command": commands[-1].tolist(), "profile": str(args.calibration_profile)})
    if not args.read_state and not args.execute:
        return
    module = _driver(args.driver)
    hand = module.LinkerHandO30Controller(hand_type="right", comm_type="libcanbus", channel=int(args.channel), canfd_device=0)
    try:
        if not hand.is_connected or tuple(hand.joint_names) != O30_ACTIVE_JOINT_NAMES:
            raise RuntimeError("O30 HOP connection or 20-axis joint order verification failed")
        current = hand.get_current_position()
        if current is None:
            raise TimeoutError("O30 feedback unavailable")
        print({"current_position_0_255": current, "temperature": hand.get_temperature()})
        if not args.execute:
            return
        # Reading feedback must remain passive.  Only execution enables the hand,
        # and the first write immediately replaces any stale controller target.
        if not hand.setup():
            raise RuntimeError("O30 HOP setup failed")
        if not hand.set_target_position(list(current)):
            raise RuntimeError("O30 rejected the initial hold command")
        time.sleep(0.05)
        current = hand.get_current_position()
        if current is None:
            raise TimeoutError("O30 feedback unavailable after setup")
        failures = 0
        target_commands = commands if args.bundle is not None or args.all_frames else commands[-1:]
        for target in target_commands:
            for command in _bounded_commands(np.asarray(current), target, args.max_step_counts):
                if not hand.set_target_position(command.tolist()):
                    raise RuntimeError("O30 rejected a bounded command")
                time.sleep(1.0 / args.fps)
                feedback = hand.get_current_position()
                if feedback is None:
                    raise TimeoutError("O30 feedback disappeared during execution")
                current = np.asarray(feedback, dtype=np.uint8)
                failures = failures + 1 if np.max(np.abs(current.astype(int) - command.astype(int))) > profile.feedback_tolerance_counts else 0
                if failures >= 3:
                    raise RuntimeError("O30 feedback lag exceeded calibration profile tolerance")
    finally:
        if args.execute and "current" in locals() and current is not None:
            try:
                hand.set_target_position(np.asarray(current, dtype=np.uint8).tolist())
                time.sleep(0.05)
            except Exception:
                pass
        hand.close()


if __name__ == "__main__":
    main()
