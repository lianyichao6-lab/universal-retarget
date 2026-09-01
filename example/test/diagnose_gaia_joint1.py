"""Inspect GaiaHand20 lateral retarget targets without connecting to the hand.

This is deliberately separate from ``example/teleop_real.py``.  It reads Pico 4
tracking data, runs the normal Gaia retargeter, and prints the five ``joint_1``
targets.  It never imports HandSDK, opens the Gaia serial port, enables motors,
or sends motor commands.
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_ROOT.parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydexretarget import Retargeter
from input.pico4 import Pico4

FINGERS = ("thumb", "index", "middle", "ring", "little")


def load_joint1_limits(hand_side: str) -> dict[str, tuple[float, float]]:
    urdf_path = (
        PROJECT_ROOT
        / "assets"
        / "gaia_hand20"
        / f"gaiahand20_{hand_side}.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for finger in FINGERS:
        joint_name = f"{hand_side}_{finger}_joint_1"
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None or joint.find("limit") is None:
            raise ValueError(f"Missing URDF limit for {joint_name}")
        limit = joint.find("limit")
        limits[finger] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print GaiaHand20 joint_1 retarget targets without connecting to "
            "or controlling the real hand"
        )
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--config",
        default="config/adaptive/pico4/pico4_gaia_hand20.yaml",
        help="Config path relative to example/",
    )
    parser.add_argument("--pico4-mode", choices=("relay", "direct"), default="relay")
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    parser.add_argument(
        "--print-hz", type=float, default=2.0,
        help="Console print frequency (default: 2 Hz)",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Stop after this many seconds; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--joint1-scale", type=float, nargs=5, default=[1.0] * 5,
        metavar=("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"),
        help="What-if scale for the five targets; diagnostics only",
    )
    parser.add_argument(
        "--joint1-offset-deg", type=float, nargs=5, default=[0.0] * 5,
        metavar=("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"),
        help="What-if offsets in degrees after scaling; diagnostics only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_hz <= 0:
        raise ValueError("--print-hz must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")

    scales = np.asarray(args.joint1_scale, dtype=np.float64)
    offsets = np.deg2rad(np.asarray(args.joint1_offset_deg, dtype=np.float64))
    if np.any(scales < 0) or not np.all(np.isfinite(scales)):
        raise ValueError("--joint1-scale must contain five finite non-negative values")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("--joint1-offset-deg must contain five finite values")

    config_path = EXAMPLE_ROOT / args.config
    retargeter = Retargeter.from_yaml(str(config_path), args.hand)
    joint_names = list(retargeter.optimizer.robot.dof_joint_names)
    joint_indices = [
        joint_names.index(f"{args.hand}_{finger}_joint_1") for finger in FINGERS
    ]
    limits = load_joint1_limits(args.hand)

    pico = Pico4(
        mode=args.pico4_mode,
        relay_host=args.pico4_relay_host,
        relay_port=args.pico4_relay_port,
        port=args.pico4_port,
        broadcast_port=args.pico4_broadcast_port,
    )

    print("Gaia joint_1 diagnostic only; HandSDK and real motors are not used.")
    print(f"Config: {config_path}")
    print("Order: thumb, index, middle, ring, little")
    print("Format: retarget -> what-if calibrated target (degrees); ! = URDF limit")

    start = time.monotonic()
    next_print = start
    print_period = 1.0 / args.print_hz
    sample_count = 0

    try:
        while args.duration == 0 or time.monotonic() - start < args.duration:
            fingers_data = pico.get_fingers_data()
            keypoints = fingers_data.get(f"{args.hand}_fingers")
            if keypoints is None or np.allclose(keypoints, 0):
                continue

            qpos = retargeter.retarget(keypoints)
            sample_count += 1
            now = time.monotonic()
            if now < next_print:
                continue

            raw = np.asarray(qpos, dtype=np.float64)[joint_indices]
            calibrated = offsets + scales * raw
            fields = []
            for i, finger in enumerate(FINGERS):
                lower, upper = limits[finger]
                clipped = float(np.clip(calibrated[i], lower, upper))
                at_limit = np.isclose(clipped, lower, atol=1e-4) or np.isclose(
                    clipped, upper, atol=1e-4
                )
                marker = "!" if at_limit else ""
                fields.append(
                    f"{finger}={np.rad2deg(raw[i]):+5.1f}"
                    f"->{np.rad2deg(clipped):+5.1f}{marker}"
                )
            print("  ".join(fields))
            next_print = now + print_period
    except KeyboardInterrupt:
        pass

    elapsed = max(time.monotonic() - start, 1e-9)
    print(f"Stopped. Retargeted {sample_count} frames ({sample_count / elapsed:.1f} Hz).")


if __name__ == "__main__":
    main()
