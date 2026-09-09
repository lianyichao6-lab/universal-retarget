#!/usr/bin/env python3
"""Create or safely jog an O30 HOP calibration profile.

This tool never searches mechanical endpoints.  Operators record safe open,
neutral and closed command values after observing one joint at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco

from anydexretarget.hand_contract import O30_ACTIVE_JOINT_NAMES
from anydexretarget.o30_hardware_profile import nominal_profile


MODEL_PATH = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _driver(path: Path):
    spec = importlib.util.spec_from_file_location("o30_hop", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HOP driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-uid", default="unbound")
    parser.add_argument("--set-knot", action="append", nargs=7, metavar=("JOINT", "QLOW", "QMID", "QHIGH", "ULOW", "UMID", "UHIGH"), help="Record one observed joint mapping; repeat for each joint")
    parser.add_argument("--feedback-tolerance", type=int, default=8)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--read-state", action="store_true")
    parser.add_argument("--jog", choices=O30_ACTIVE_JOINT_NAMES)
    parser.add_argument("--delta-counts", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--channel", default="0")
    args = parser.parse_args()
    profile = nominal_profile(mujoco.MjModel.from_xml_path(str(MODEL_PATH)), device_uid=args.device_uid)
    if args.feedback_tolerance <= 0:
        parser.error("--feedback-tolerance must be positive")
    q_knots, u_knots = profile.qpos_knots_rad.copy(), profile.command_knots_u8.copy()
    for item in args.set_knot or []:
        joint, *values = item
        index = O30_ACTIVE_JOINT_NAMES.index(joint)
        q_knots[index] = [float(value) for value in values[:3]]
        u_knots[index] = [float(value) for value in values[3:]]
    profile = replace(profile, qpos_knots_rad=q_knots, command_knots_u8=u_knots, feedback_tolerance_counts=args.feedback_tolerance)
    profile.save(args.output)
    print(f"Nominal O30 calibration template written: {args.output}")
    if not args.read_state and args.jog is None:
        return
    if args.driver is None:
        parser.error("--driver is required for --read-state or --jog")
    if args.jog is not None and (not args.execute or args.confirm != "O30_CALIBRATION_CLEAR"):
        parser.error("--jog requires --execute --confirm O30_CALIBRATION_CLEAR")
    module = _driver(args.driver)
    hand = module.LinkerHandO30Controller(hand_type="right", comm_type="libcanbus", channel=int(args.channel), canfd_device=0)
    try:
        if not hand.is_connected or tuple(hand.joint_names) != O30_ACTIVE_JOINT_NAMES:
            raise RuntimeError("O30 HOP connection or joint order check failed")
        current = hand.get_current_position()
        if current is None:
            raise TimeoutError("O30 feedback unavailable")
        print({"joint_names": list(hand.joint_names), "current_position_0_255": current})
        if args.jog is None:
            return
        if not hand.setup():
            raise RuntimeError("O30 HOP setup failed")
        if not hand.set_target_position(list(current)):
            raise RuntimeError("O30 rejected the initial hold command")
        time.sleep(0.05)
        if args.jog is not None:
            index = hand.joint_names.index(args.jog)
            target = max(0, min(255, int(current[index]) + args.delta_counts))
            if not hand.set_joint_position(args.jog, target):
                raise RuntimeError("O30 rejected single-joint calibration jog")
            time.sleep(0.05)
            setting = hand.get_target_position()
            if setting is None:
                raise TimeoutError("O30 target-position register did not respond after jog")
            accepted = int(setting[index])
            print({
                "jogged_joint": args.jog,
                "requested_0_255": target,
                "accepted_0_255": accepted,
            })
            if accepted != target:
                raise RuntimeError(
                    f"O30 {args.jog} target register contains {accepted}, expected {target}"
                )
            deadline = time.monotonic() + 2.0
            observed = int(current[index])
            while time.monotonic() < deadline:
                time.sleep(0.05)
                feedback = hand.get_current_position()
                if feedback is None:
                    continue
                current = feedback
                observed = int(feedback[index])
                if abs(observed - target) <= args.feedback_tolerance:
                    break
            print({"jogged_joint": args.jog, "target_0_255": target, "observed_0_255": observed})
            if abs(observed - target) > args.feedback_tolerance:
                raise TimeoutError(
                    f"O30 {args.jog} feedback remained outside +\/-"
                    f"{args.feedback_tolerance} counts for 2 seconds"
                )
    finally:
        if args.jog is not None and "current" in locals() and current is not None:
            try:
                hand.set_target_position(list(current))
                time.sleep(0.05)
            except Exception:
                pass
        hand.close()


if __name__ == "__main__":
    main()
