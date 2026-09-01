#!/usr/bin/env python3
"""Safely inspect or incrementally command a right LinkerHand L25 over CAN.

The default mode is offline validation.  ``--read-state`` opens the vendor SDK
only after CAN is already UP and performs no motion.  Actual motion requires
both ``--hardware`` and the literal ``--confirm L25_RIGHT_CLEAR``.  Hardware
commands are ramped from the reported 25-channel state and are deliberately
limited to a small number of SDK units per invocation.

This utility is for empty-space calibration only.  It does not establish an
object grasp, arm pose, collision clearance, or force closure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK_PACKAGE = os.environ.get("LINKERHAND_SDK_PACKAGE", "")


def _load_local_hardware_adapter():
    """Avoid importing AnyDex optimizer dependencies in the SDK process."""
    adapter_path = ROOT / "anydexretarget" / "hardware_adapter.py"
    spec = importlib.util.spec_from_file_location("local_l25_hardware_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load hardware adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HARDWARE_ADAPTER = _load_local_hardware_adapter()
INDEPENDENT_TO_SDK = _HARDWARE_ADAPTER.INDEPENDENT_TO_SDK
L25HardwareAdapter = _HARDWARE_ADAPTER.L25HardwareAdapter
L25_QPOS_JOINTS = _HARDWARE_ADAPTER.L25_QPOS_JOINTS
RANGES = _HARDWARE_ADAPTER.RANGES

CONFIRMATION = "L25_RIGHT_CLEAR"


def _can_is_up(interface: str) -> bool:
    result = subprocess.run(
        ["ip", "link", "show", interface], text=True, capture_output=True, check=False
    )
    return result.returncode == 0 and "state UP" in result.stdout


def _load_trajectory(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    frames: list[np.ndarray] = []
    for frame, record in enumerate(records):
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError(f"trajectory frame {frame} has no target")
        qpos = np.asarray(record["target"], dtype=np.float64)
        if qpos.shape != (21,) or not np.isfinite(qpos).all():
            raise ValueError(f"trajectory frame {frame} has invalid L25 qpos")
        frames.append(qpos)
    return frames


def _validate_qpos(qpos: np.ndarray) -> None:
    by_name = dict(zip(L25_QPOS_JOINTS, qpos))
    violations: list[str] = []
    for name, channel in INDEPENDENT_TO_SDK.items():
        lo, hi, _, _ = RANGES[channel]
        value = float(by_name[name])
        if value < lo - 1e-8 or value > hi + 1e-8:
            violations.append(f"{name}={value:.5f} outside [{lo:.5f}, {hi:.5f}]")
    if violations:
        raise ValueError("refusing out-of-limit qpos: " + "; ".join(violations))


def _load_vendor_api(sdk_package: Path):
    module_path = sdk_package / "LinkerHand" / "linker_hand_api.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"L25 vendor SDK not found: {module_path}")
    spec = importlib.util.spec_from_file_location("l25_vendor_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load L25 vendor SDK")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LinkerHandApi


def _finite_state(api) -> np.ndarray:
    state = np.asarray(api.get_state(), dtype=np.float64)
    if state.shape != (25,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid L25 SDK state: shape={state.shape}")
    if np.any(state < 0) or np.any(state > 255):
        raise RuntimeError("L25 SDK reported a state outside 0..255")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Target frame for a single incremental move")
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "--sdk-package", type=Path,
        default=Path(DEFAULT_SDK_PACKAGE) if DEFAULT_SDK_PACKAGE else None,
        help="Directory containing LinkerHand/linker_hand_api.py; may also be set with LINKERHAND_SDK_PACKAGE",
    )
    parser.add_argument("--read-state", action="store_true", help="Open SDK and report state; never moves")
    parser.add_argument("--hardware", action="store_true", help="Permit one bounded real command")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRMATION!r} for --hardware")
    parser.add_argument(
        "--channels", default="",
        help="Comma-separated SDK channels allowed to move. Required for --hardware; ROOT2/TIP partners are included automatically.",
    )
    parser.add_argument("--replay-target", action="store_true", help="Ramp all SDK channels to the selected frame at the bounded rate")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Command rate for --replay-target, 2..20 Hz")
    parser.add_argument("--max-step", type=int, default=2, help="Maximum 0..255 change per channel, 1..5")
    parser.add_argument("--speed", type=int, default=30, help="Vendor speed for the five fingers, 10..80")
    parser.add_argument("--torque", type=int, default=40, help="Vendor torque for the five fingers, 10..80")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if not (1 <= args.max_step <= 5):
        parser.error("--max-step must be in 1..5")
    if not (2.0 <= args.rate_hz <= 20.0):
        parser.error("--rate-hz must be in 2..20")
    if not (10 <= args.speed <= 80 and 10 <= args.torque <= 80):
        parser.error("--speed and --torque must be in 10..80 for calibration")
    if args.hardware and args.confirm != CONFIRMATION:
        parser.error(f"real motion requires --confirm {CONFIRMATION}")
    if args.hardware and not args.read_state:
        parser.error("real motion requires --read-state so feedback is verified first")
    selected_channels: set[int] = set()
    if args.channels:
        try:
            selected_channels = {int(value) for value in args.channels.split(",")}
        except ValueError as exc:
            parser.error(f"--channels must be comma-separated integers: {exc}")
        if any(channel < 0 or channel >= 25 for channel in selected_channels):
            parser.error("--channels must be within 0..24")
        for channel in tuple(selected_channels):
            if 15 <= channel <= 19:
                selected_channels.add(channel + 5)
            elif 20 <= channel <= 24:
                selected_channels.add(channel - 5)
    if args.replay_target:
        selected_channels = set(range(25))
    if args.hardware and not selected_channels:
        parser.error("real motion requires an explicit --channels allowlist or --replay-target")
    if args.read_state and args.sdk_package is None:
        parser.error("--read-state requires --sdk-package or LINKERHAND_SDK_PACKAGE")

    frames = _load_trajectory(args.trajectory)
    if not 0 <= args.frame < len(frames):
        parser.error(f"--frame must be in 0..{len(frames) - 1}")
    qpos = frames[args.frame]
    _validate_qpos(qpos)
    command = L25HardwareAdapter().qpos_to_command(qpos).values.astype(np.int64)
    report = {
        "trajectory": str(args.trajectory.resolve()),
        "frame": int(args.frame),
        "hardware_requested": bool(args.hardware),
        "qpos_finite": bool(np.isfinite(qpos).all()),
        "target_command_0_255": command.tolist(),
        "target_range": [int(command.min()), int(command.max())],
    }

    if not args.read_state:
        print("Offline validation passed; no SDK opened and no command sent.")
        print(f"target range={report['target_range']}; use --read-state for a hardware read-only preflight")
    else:
        if not _can_is_up(args.can):
            raise RuntimeError(
                f"{args.can} is not UP. Enable it manually at 1 Mbps, then rerun; this tool will not alter CAN configuration."
            )
        api = None
        try:
            api = _load_vendor_api(args.sdk_package)(hand_type="right", hand_joint="L25", modbus="None", can=args.can)
            state = _finite_state(api)
            report["state_command_0_255"] = state.astype(int).tolist()
            report["state_target_max_delta"] = int(np.max(np.abs(command - state)))
            print("L25 read-only preflight passed")
            print(f"state range=[{int(state.min())}, {int(state.max())}], target max delta={report['state_target_max_delta']}")
            if args.hardware:
                selected = np.asarray(sorted(selected_channels), dtype=np.int64)
                next_command = state.astype(np.int64).copy()
                api.set_speed([args.speed] * 5)
                api.set_torque([args.torque] * 5)
                if args.replay_target:
                    steps = 0
                    while not np.array_equal(next_command[selected], command[selected]):
                        next_command[selected] += np.clip(
                            command[selected] - next_command[selected], -args.max_step, args.max_step
                        )
                        if np.any(next_command < 0) or np.any(next_command > 255):
                            raise RuntimeError("bounded L25 command is outside 0..255")
                        api.finger_move(next_command.tolist())
                        steps += 1
                        if steps > 60:
                            raise RuntimeError("replay exceeded the 60-step safety limit")
                        time.sleep(1.0 / args.rate_hz)
                    report["replay_steps"] = steps
                    print(f"Replayed one L25 target in {steps} bounded steps.")
                else:
                    next_command[selected] += np.clip(
                        command[selected] - next_command[selected], -args.max_step, args.max_step
                    )
                    if np.any(next_command < 0) or np.any(next_command > 255):
                        raise RuntimeError("bounded L25 command is outside 0..255")
                    api.finger_move(next_command.tolist())
                    print("Sent exactly one bounded L25 calibration increment.")
                report["sent_command_0_255"] = next_command.tolist()
                report["max_step"] = int(args.max_step)
                report["selected_channels"] = sorted(selected_channels)
                report["speed"] = int(args.speed)
                report["torque"] = int(args.torque)
                time.sleep(0.25)
                after = _finite_state(api)
                report["state_after_0_255"] = after.astype(int).tolist()
                print(f"post-command state range=[{int(after.min())}, {int(after.max())}]")
        finally:
            if api is not None:
                # Vendor close_can() is broken in SDK 3.1.1 and may also bring
                # down the shared can0 interface.  Release only this process socket.
                hand = getattr(api, "hand", None)
                if hand is not None:
                    hand.running = False
                    shutdown = getattr(getattr(hand, "bus", None), "shutdown", None)
                    if callable(shutdown):
                        shutdown()

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
