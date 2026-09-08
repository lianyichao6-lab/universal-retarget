#!/usr/bin/env python3
"""Validate or play an O30 trajectory through the standalone HOP driver.

This uses the direct vendor HOP driver. It loads the vendor's direct HOP
Python driver by path, converts O30 physical-qpos radians to its documented
0..255 units, and is read-only unless --execute is explicitly confirmed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path
from types import ModuleType

import mujoco
import numpy as np

from anydexretarget.hand_contract import (
    O30_ACTIVE_JOINT_NAMES,
    O30_ACTIVE_SOURCE_NAMES,
    O30_QPOS_JOINT_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _load_targets(path: Path, all_frames: bool, frame: int) -> list[np.ndarray]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    selected = records if all_frames else [records[frame]]
    targets: list[np.ndarray] = []
    for record in selected:
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError("trajectory record lacks target")
        names = tuple(record.get("robot_joint_names", O30_QPOS_JOINT_NAMES))
        qpos = np.asarray(record["target"], dtype=np.float64).reshape(-1)
        if names != O30_QPOS_JOINT_NAMES or qpos.shape != (20,) or not np.isfinite(qpos).all():
            raise ValueError("trajectory is not a finite O30 physical-qpos trajectory")
        targets.append(qpos)
    return targets


def _qpos_to_hop_u8(qpos: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    joint_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
        for index in range(model.njnt)
    }
    values = dict(zip(O30_QPOS_JOINT_NAMES, qpos, strict=True))
    command: list[int] = []
    for source_name in O30_ACTIVE_SOURCE_NAMES:
        joint_id = joint_ids.get(source_name)
        if joint_id is None:
            raise ValueError(f"O30 URDF lacks {source_name}")
        lower, upper = model.jnt_range[joint_id]
        if upper <= lower:
            raise ValueError(f"invalid O30 joint range for {source_name}")
        normalized = (values[source_name] - lower) / (upper - lower)
        command.append(int(np.rint(np.clip(normalized, 0.0, 1.0) * 255.0)))
    return np.asarray(command, dtype=np.uint8)


def _load_driver(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("anydex_o30_hop_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HOP driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "LinkerHandO30Controller"):
        raise ImportError(f"No LinkerHandO30Controller in {path}")
    return module


def _connect(args: argparse.Namespace):
    if args.driver is None:
        raise ValueError("--driver is required for --read-state or --execute")
    module = _load_driver(args.driver)
    channel: int | str = int(args.channel) if args.comm_type == "libcanbus" else args.channel
    hand = module.LinkerHandO30Controller(
        hand_type=args.side,
        canfd_device=args.canfd_device,
        comm_type=args.comm_type,
        channel=channel,
    )
    if not hand.is_connected:
        hand.close()
        raise ConnectionError("O30 HOP driver did not connect to the requested transport/channel")
    names = tuple(getattr(hand, "joint_names", ()))
    if names != O30_ACTIVE_JOINT_NAMES:
        hand.close()
        raise RuntimeError(f"unexpected O30 HOP joint order: {names}")
    return hand


def _ramp_commands(start: object, target: object, frames: int) -> list[np.ndarray]:
    """Return bounded integer HOP commands from feedback to a first target."""
    source = np.asarray(start, dtype=np.float64).reshape(-1)
    destination = np.asarray(target, dtype=np.float64).reshape(-1)
    if source.shape != (20,) or destination.shape != (20,):
        raise ValueError("O30 ramp endpoints must each contain 20 values")
    if frames < 1:
        raise ValueError("O30 ramp frame count must be positive")
    if np.any(source < 0) or np.any(source > 255) or np.any(destination < 0) or np.any(destination > 255):
        raise ValueError("O30 ramp endpoints must stay within [0, 255]")
    return [
        np.rint(source + (index / frames) * (destination - source)).astype(np.uint8)
        for index in range(1, frames + 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--comm-type", choices=("libcanbus", "socketcan"), default="libcanbus")
    parser.add_argument("--canfd-device", type=int, default=0)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--read-state", action="store_true")
    parser.add_argument("--ramp-from-current", action="store_true", help="Ramp feedback to the trajectory first frame before executing")
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.fps <= 0 or args.ramp_seconds <= 0:
        parser.error("--fps and --ramp-seconds must be positive")
    if args.execute and args.confirm != "O30_HAND_CLEAR":
        parser.error("--execute requires --confirm O30_HAND_CLEAR")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    qposes = _load_targets(args.trajectory, args.all_frames, args.frame)
    commands = [_qpos_to_hop_u8(qpos, model) for qpos in qposes]
    print(json.dumps({
        "planning_only": not args.execute,
        "frame_count": len(commands),
        "fps": args.fps if args.all_frames else None,
        "hop_joint_names": list(O30_ACTIVE_JOINT_NAMES),
        "first_command_0_255": commands[0].tolist(),
        "last_command_0_255": commands[-1].tolist(),
        "ramp_from_current": args.ramp_from_current if args.execute else False,
        "ramp_seconds": args.ramp_seconds if args.execute and args.ramp_from_current else None,
    }, indent=2))
    if not args.read_state and not args.execute:
        return
    hand = _connect(args)
    try:
        if args.read_state:
            current = hand.get_current_position()
            if current is None:
                raise TimeoutError("O30 did not return current joint positions")
            print(json.dumps({"current_position_0_255": current}, indent=2))
        if args.execute:
            if args.ramp_from_current:
                current = hand.get_current_position()
                if current is None:
                    raise TimeoutError("O30 did not return current joint positions for ramp")
                ramp_frames = max(1, int(np.ceil(args.ramp_seconds * args.fps)))
                for command in _ramp_commands(current, commands[0], ramp_frames):
                    if not hand.set_target_position(command.tolist()):
                        raise RuntimeError("O30 rejected a feedback-ramp command")
                    time.sleep(1.0 / args.fps)
            if not hand.setup():
                raise RuntimeError("O30 HOP setup failed")
            period = 1.0 / args.fps if args.all_frames else 0.0
            for index, command in enumerate(commands):
                started = time.monotonic()
                if not hand.set_target_position(command.tolist()):
                    raise RuntimeError(f"O30 rejected command frame {index}")
                if args.all_frames and index + 1 < len(commands):
                    time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        hand.close()


if __name__ == "__main__":
    main()
