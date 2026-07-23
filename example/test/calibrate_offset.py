#!/usr/bin/env python3
"""Calibrate ``wrist_offset_cm`` and ``thumb_offset_cm`` for AnyDexRetarget.

The script was ported from ``wuji-retargeting/example/calibrate_offset.py``.
It compares transformed hand MCP/CMC landmarks with the configured robot's
neutral-pose finger-root positions. Paths are resolved relative to ``example/``.

Examples:
    # Replay a trusted recording
    python example/test/calibrate_offset.py \
        --play data/avp1.pkl --trust-pkl --hand left \
        --config config/adaptive/avp/avp_wuji_hand.yaml

    # Capture from the default camera and save the samples
    python example/test/calibrate_offset.py --input camera --hand right \
        --config config/adaptive/mediapipe/mediapipe_wuji_hand.yaml \
        --dump-pkl data/calibration_right.pkl

    # Capture from the bundled video input
    python example/test/calibrate_offset.py --input video \
        --source data/right.mp4 --hand right
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations

MP_MCP_INDICES = [1, 5, 9, 13, 17]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def resolve_example_path(path_value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to ``example/``."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = EXAMPLE_ROOT / path
    return path.resolve()


def get_robot_mcp_positions(retargeter: Retargeter):
    """Return neutral-pose robot finger roots relative to the robot origin.

    Returns:
        ``(finger_indices, positions)`` where ``finger_indices`` contains the
        corresponding MediaPipe finger numbers (thumb=0 ... pinky=4), and
        ``positions`` has shape ``(N, 3)``.
    """
    optimizer = retargeter.optimizer
    robot = optimizer.robot

    neutral_qpos = getattr(optimizer, "neutral_qpos", None)
    if neutral_qpos is None:
        neutral_qpos = np.zeros(robot.model.nq, dtype=np.float64)
    else:
        neutral_qpos = np.asarray(neutral_qpos, dtype=np.float64).copy()
        if neutral_qpos.shape != (robot.model.nq,):
            raise ValueError(
                "neutral_qpos length does not match the robot model: "
                f"{neutral_qpos.shape} vs ({robot.model.nq},)"
            )

    robot.compute_forward_kinematics(neutral_qpos)
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.get_link_pose(origin_id)[:3, 3].copy()

    finger_indices: list[int] = []
    positions: list[np.ndarray] = []
    for local_i, mp_finger_i in enumerate(optimizer.mp_finger_indices):
        if local_i >= len(optimizer.link1_names):
            continue
        link_id = robot.get_link_index(optimizer.link1_names[local_i])
        link_pos = robot.get_link_pose(link_id)[:3, 3].copy()
        finger_indices.append(int(mp_finger_i))
        positions.append(link_pos - origin_pos)

    if not positions:
        raise RuntimeError("The configured optimizer does not expose finger-root links")
    return finger_indices, np.asarray(positions, dtype=np.float64)


def compute_mediapipe_mcp_positions(
    raw_keypoints: np.ndarray,
    hand_side: str,
    rotation_xyz: dict,
    finger_indices: list[int],
) -> np.ndarray:
    """Transform one frame and return its requested MCP/CMC positions."""
    kp = apply_mediapipe_transformations(raw_keypoints, hand_side)

    x_deg = rotation_xyz.get("x", 0.0)
    y_deg = rotation_xyz.get("y", 0.0)
    z_deg = rotation_xyz.get("z", 0.0)
    if x_deg != 0 or y_deg != 0 or z_deg != 0:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler(
            "xyz", [x_deg, y_deg, z_deg], degrees=True
        )
        kp = kp @ rot.as_matrix().T

    return kp[[MP_MCP_INDICES[i] for i in finger_indices]]


def collect_frames_from_pkl(
    pkl_path: Path, limit: int | None, trust_pkl: bool = False
) -> list[dict]:
    """Load recorded frames after explicit pickle trust confirmation."""
    if not trust_pkl:
        raise ValueError(
            "Refusing to load pickle without explicit trust. "
            "Use --trust-pkl only for files you fully trust."
        )
    with pkl_path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Expected a frame list in {pkl_path}, got {type(data).__name__}")
    data = list(data)
    print(f"Loaded {len(data)} frames from {pkl_path}")
    if limit is not None:
        data = data[:limit]
        print(f"Using first {len(data)} frames")
    return data


def _create_input_device(args: argparse.Namespace) -> Any:
    """Construct one of the input devices shipped in ``example/input``."""
    if args.input == "camera":
        from input.camera import Camera

        return Camera(camera_id=args.camera_id, show_preview=args.show_video)
    if args.input == "video":
        from input.video import Video

        video_path = resolve_example_path(args.source or "data/right.mp4")
        return Video(
            video_path=str(video_path),
            hand_side=args.hand,
            show_video=args.show_video,
            playback_speed=args.playback_speed,
            loop=True,
            depth_scale=args.depth_scale,
        )
    if args.input == "realsense":
        from input.realsense import Realsense

        return Realsense(hand_side=args.hand, show_video=args.show_video)
    if args.input == "replay":
        if not args.source:
            raise ValueError("--source is required when --input replay is used")
        if not args.trust_pkl:
            raise ValueError("--input replay loads pickle data; pass --trust-pkl after verifying it")
        from input.mediapipe_replay import MediaPipeReplay

        return MediaPipeReplay(
            record_path=str(resolve_example_path(args.source)),
            playback_speed=args.playback_speed,
            loop=True,
        )
    if args.input == "noitom":
        from input.noitom import NoitomInput

        return NoitomInput(
            local_ip=args.noitom_local_ip,
            local_port=args.noitom_local_port,
            server_ip=args.noitom_server_ip,
            server_port=args.noitom_server_port,
        )
    if args.input == "quest3":
        from input.quest3 import Quest3

        return Quest3(port=args.quest3_port, protocol=args.quest3_protocol)
    if args.input == "avp":
        from input.visionpro import VisionPro

        return VisionPro(ip=args.avp_ip)
    if args.input == "pico4":
        from input.pico4 import Pico4

        return Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
    raise ValueError(f"Unsupported input type: {args.input}")


def _close_input_device(device: Any) -> None:
    for method_name in ("cleanup", "release", "stop", "close"):
        method = getattr(device, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # cleanup should not hide calibration output
                print(f"Warning: {method_name}() failed: {exc}")
            break


def collect_frames_live(args: argparse.Namespace) -> list[dict]:
    """Sample ``num_frames`` valid live frames from the selected device."""
    if args.rate_hz <= 0:
        raise ValueError(f"--rate-hz must be > 0, got {args.rate_hz}")
    if args.num_frames <= 0:
        raise ValueError(f"--num-frames must be > 0, got {args.num_frames}")

    device = _create_input_device(args)
    hand_key = f"{args.hand}_fingers"
    period = 1.0 / args.rate_hz
    frames: list[dict] = []
    skipped = 0
    try:
        print(
            f"\nPlace your {args.hand} hand in the calibration pose "
            "(palm flat, fingers relaxed and straight)."
        )
        if not args.no_wait:
            input(f"Press Enter to capture {args.num_frames} valid frames... ")
        else:
            print(f"Capturing {args.num_frames} valid frames immediately...")

        deadline = time.monotonic() + args.timeout
        while len(frames) < args.num_frames:
            loop_start = time.monotonic()
            frame = device.get_fingers_data()
            raw_kp = frame.get(hand_key)
            if raw_kp is None or np.asarray(raw_kp).shape != (21, 3) or np.allclose(raw_kp, 0):
                skipped += 1
            else:
                frames.append(
                    {
                        "t": time.time(),
                        "left_fingers": np.asarray(
                            frame.get("left_fingers", np.zeros((21, 3))),
                            dtype=np.float32,
                        ).copy(),
                        "right_fingers": np.asarray(
                            frame.get("right_fingers", np.zeros((21, 3))),
                            dtype=np.float32,
                        ).copy(),
                    }
                )
                if len(frames) == 1 or len(frames) % 10 == 0 or len(frames) == args.num_frames:
                    print(f"  captured {len(frames)}/{args.num_frames}", end="\r", flush=True)

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {args.timeout:.1f}s with "
                    f"{len(frames)}/{args.num_frames} valid frames"
                )
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        _close_input_device(device)

    print(f"\nCaptured {len(frames)} valid frames; skipped {skipped} invalid reads")
    return frames


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate AnyDexRetarget wrist/thumb offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/adaptive/mediapipe/mediapipe_wuji_hand.yaml",
        help="Retarget YAML path, absolute or relative to example/",
    )
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument(
        "--input",
        default="camera",
        choices=["camera", "video", "realsense", "replay", "noitom", "quest3", "avp", "pico4"],
        help="Live/input-device mode used when --play is not supplied",
    )
    parser.add_argument(
        "--play",
        metavar="FILE",
        help="Load a recorded frame list (.pkl), relative to example/",
    )
    parser.add_argument(
        "--source",
        help="Input source path for video/replay (relative to example/)",
    )
    parser.add_argument("--frames", type=int, default=None, help="Use only the first N --play frames")
    parser.add_argument("--trust-pkl", action="store_true", help="Confirm that pickle input is trusted")
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-wait", action="store_true", help="Start live sampling without waiting for Enter")
    parser.add_argument("--dump-pkl", help="Save captured raw frames relative to example/")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--noitom-local-ip", default="192.168.5.25")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", choices=["udp", "tcp"], default="udp")
    parser.add_argument("--avp-ip", default="192.168.50.127")
    parser.add_argument("--pico4-mode", choices=["relay", "direct"], default="relay")
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = resolve_example_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Retarget config not found: {config_path}")

    if args.play:
        data = collect_frames_from_pkl(
            resolve_example_path(args.play), args.frames, args.trust_pkl
        )
    else:
        data = collect_frames_live(args)
        if args.dump_pkl:
            dump_path = resolve_example_path(args.dump_pkl)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("wb") as file:
                pickle.dump(data, file)
            print(f"Raw calibration frames saved to {dump_path}")

    if not data:
        print("ERROR: no frames collected")
        return 1

    retargeter = Retargeter.from_yaml(str(config_path), hand_side=args.hand)
    finger_indices, robot_mcp = get_robot_mcp_positions(retargeter)

    print("\nRobot finger-root positions at neutral qpos (meters):")
    for finger_i, pos in zip(finger_indices, robot_mcp):
        print(f"  {FINGER_NAMES[finger_i]:7s}: [{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]")

    hand_key = f"{args.hand}_fingers"
    diffs: list[np.ndarray] = []
    skipped = 0
    for frame in data:
        raw_kp = frame.get(hand_key) if isinstance(frame, dict) else None
        if raw_kp is None or np.asarray(raw_kp).shape != (21, 3) or np.allclose(raw_kp, 0):
            skipped += 1
            continue
        mp_mcp = compute_mediapipe_mcp_positions(
            np.asarray(raw_kp), args.hand, retargeter.rotation_xyz, finger_indices
        )
        diffs.append(robot_mcp - mp_mcp)

    if not diffs:
        print("ERROR: no valid hand frames found")
        return 1

    diffs_array = np.asarray(diffs)
    per_finger = np.median(diffs_array, axis=0)
    print(f"\nProcessed {len(diffs)} frames (skipped {skipped})")
    print("Median per-finger root offset (robot - input, meters):")
    for finger_i, delta in zip(finger_indices, per_finger):
        print(f"  {FINGER_NAMES[finger_i]:7s}: [{delta[0]:+.5f}, {delta[1]:+.5f}, {delta[2]:+.5f}]")

    thumb_rows = [i for i, finger_i in enumerate(finger_indices) if finger_i == 0]
    other_rows = [i for i, finger_i in enumerate(finger_indices) if finger_i != 0]
    if not thumb_rows or not other_rows:
        raise RuntimeError("Calibration needs one thumb and at least one non-thumb finger")

    thumb_offset_m = np.median(per_finger[thumb_rows], axis=0)
    wrist_offset_m = np.median(per_finger[other_rows], axis=0)
    thumb_offset_cm = thumb_offset_m * 100.0
    wrist_offset_cm = wrist_offset_m * 100.0

    print(f"\n{'=' * 64}")
    print("Recommended values under the YAML 'retarget:' section:")
    print(f"{'=' * 64}")
    print(
        "  wrist_offset_cm: "
        f"[{wrist_offset_cm[0]:.3f}, {wrist_offset_cm[1]:.3f}, {wrist_offset_cm[2]:.3f}]"
    )
    print(
        "  thumb_offset_cm: "
        f"[{thumb_offset_cm[0]:.3f}, {thumb_offset_cm[1]:.3f}, {thumb_offset_cm[2]:.3f}]"
    )
    print(f"{'=' * 64}")

    print("\nPer-finger residual after applying the group offset (cm):")
    for finger_i, delta in zip(finger_indices, per_finger):
        group_offset = thumb_offset_m if finger_i == 0 else wrist_offset_m
        residual = (delta - group_offset) * 100.0
        print(f"  {FINGER_NAMES[finger_i]:7s}: [{residual[0]:+.3f}, {residual[1]:+.3f}, {residual[2]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
