"""Inspect or reset GaiaHand20 hardware joint zero positions.

This utility is intentionally separate from ``example/teleop_real.py``.
Resetting a hardware zero changes the motor controller's interpretation of all
future joint commands.  The reset action therefore requires an exact, explicit
confirmation after the hand has been manually aligned with the URDF zero pose.

It does not connect to Pico tracking and never starts teleoperation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4")
RESET_CONFIRMATION = "RESET CURRENT GAIA POSE AS ZERO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or reset GaiaHand20 hardware zero positions"
    )
    parser.add_argument(
        "--action",
        choices=("inspect", "reset-zero"),
        default="inspect",
        help="inspect is read-only; reset-zero writes the current physical pose as zero",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--use-slcan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--has-main-board",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finger",
        choices=("all",) + FINGER_NAMES,
        default="all",
        help="joint selection for reset-zero",
    )
    parser.add_argument(
        "--joint",
        choices=("all",) + JOINT_NAMES,
        default="all",
        help="joint selection for reset-zero; joint_4 exists only on the thumb",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.8,
        help="motor-angle query timeout in seconds",
    )
    parser.add_argument(
        "--broadcast-read",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="query all motor angles through one broadcast request",
    )
    return parser.parse_args()


def print_gaia20_mapping(mapping_manager) -> None:
    print("GaiaHand20 SDK mapping:")
    for finger_name in FINGER_NAMES:
        finger = mapping_manager._finger_enum[finger_name]
        joint_count = mapping_manager.manager.get_finger_joint_count(finger)
        fields = []
        for joint_index in range(joint_count):
            joint = mapping_manager._joint_enum[joint_index]
            motor_id = mapping_manager.manager.get_motor_id(finger, joint)
            fields.append(f"joint_{joint_index + 1}=motor{motor_id}")
        print(f"  {finger_name:6s}: " + ", ".join(fields))


class MappingView:
    def __init__(self, hand_type, hand_side, finger_type, joint_type, manager_cls):
        self.manager = manager_cls(hand_type, hand_side)
        self._finger_enum = {
            "thumb": finger_type.THUMB,
            "index": finger_type.INDEX,
            "middle": finger_type.MIDDLE,
            "ring": finger_type.RING,
            "little": finger_type.LITTLE,
        }
        self._joint_enum = (
            joint_type.JOINT_1,
            joint_type.JOINT_2,
            joint_type.JOINT_3,
            joint_type.JOINT_4,
        )


def print_motor_angles(hand, mapping: MappingView, timeout: float, use_broadcast: bool) -> None:
    """Read raw motor angles without treating a timeout as motor failure."""
    print("\nReading raw motor angles (SDK values; read timeout is not a motor fault)...")
    try:
        angles = hand.get_all_motor_angle(
            sync=True,
            timeout=timeout,
            use_broadcast=use_broadcast,
        )
    except Exception as exc:
        print(f"  Angle query failed or timed out: {exc}")
        return

    if not angles:
        print("  No angle response received. This does not prove that a motor is broken.")
        return

    for motor_id in sorted(angles):
        finger, joint = mapping.manager.get_finger_joint_from_motor_id(int(motor_id))
        value = float(angles[motor_id])
        # SDK documentation labels this low-level API as degrees. Print the raw
        # value as the authority, and also a radian interpretation for spotting
        # version-specific documentation inconsistencies.
        print(
            f"  motor {int(motor_id):2d}  {finger.value:6s} {joint.value:5s}: "
            f"raw={value:+10.5f}  raw-as-rad={math.degrees(value):+9.3f} deg"
        )


def resolve_selection(args, mapping: MappingView):
    if args.finger == "all":
        if args.joint != "all":
            raise ValueError("--joint requires selecting one --finger")
        return None, None, "all 16 joints"

    finger = mapping._finger_enum[args.finger]
    if args.joint == "all":
        return finger, None, f"all joints of {args.finger}"

    joint_index = JOINT_NAMES.index(args.joint)
    if joint_index == 3 and args.finger != "thumb":
        raise ValueError("joint_4 exists only on the GaiaHand20 thumb")
    joint = mapping._joint_enum[joint_index]
    motor_id = mapping.manager.get_motor_id(finger, joint)
    return finger, joint, f"{args.finger} {args.joint} (motor {motor_id})"


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    try:
        from hand import create_hand
        from hand.gaiahand.hand_mappings import (
            FingerType,
            HandSide,
            HandType,
            JointType,
        )
        from hand.gaiahand.motor_mappings import MotorMappingManager
    except ImportError as exc:
        raise ImportError(
            "Run this script in the anydex environment with HandSDK installed"
        ) from exc

    hand_side = HandSide.RIGHT if args.hand == "right" else HandSide.LEFT
    mapping = MappingView(
        HandType.GAIA_20,
        hand_side,
        FingerType,
        JointType,
        MotorMappingManager,
    )
    print_gaia20_mapping(mapping)

    hand = create_hand(
        "gaia20",
        args.hand,
        port=args.port,
        baudrate=args.baudrate,
        use_slcan=args.use_slcan,
        has_main_board=args.has_main_board,
    )
    connected = False
    try:
        if not hand.connect():
            raise ConnectionError(f"Failed to connect to GaiaHand20 at {args.port}")
        connected = True
        print(
            f"\nConnected: {args.hand} GaiaHand20 at {args.port} "
            f"@ {args.baudrate} baud"
        )

        if args.action == "inspect":
            print_motor_angles(hand, mapping, args.timeout, args.broadcast_read)
            print("\nRead-only inspection complete; no motor enable or zero reset was sent.")
            return

        finger, joint, selection = resolve_selection(args, mapping)
        print("\nWARNING: HARDWARE ZERO RESET")
        print(f"Selected: {selection}")
        print("1. Stop teleop_real.py and close HandGUI.")
        print("2. Keep motors disabled.")
        print("3. Manually align the selected physical joint(s) with the Gaia URDF zero pose.")
        print("4. Hold the hand in that exact pose while confirming.")
        print("5. A wrong pose here will make all future commands wrong and may cause collisions.")

        # Explicitly request motor disable before the user aligns the hand.
        disabled = hand.enable_all_motors_broadcast(False)
        print(f"\nDisable-all command result: {disabled}")
        typed = input(f'Type exactly "{RESET_CONFIRMATION}" to continue:\n> ').strip()
        if typed != RESET_CONFIRMATION:
            print("Confirmation did not match; zero reset cancelled.")
            return

        success = hand.hand_reset_zero(finger=finger, joint=joint)
        if not success:
            raise RuntimeError(f"HandSDK reported zero-reset failure for {selection}")
        print(f"Hardware zero reset completed for {selection}.")
        print("Motors remain disabled. Run --action inspect, then test at low speed.")
    finally:
        if connected:
            try:
                hand.enable_all_motors_broadcast(False)
            except Exception:
                pass
        try:
            hand.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
