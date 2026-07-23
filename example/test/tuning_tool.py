#!/usr/bin/env python3
"""Interactive parameter-tuning viewer for AnyDexRetarget.

This is the port of ``wuji-retargeting/example/tuning_tool.py``. It lives in
``example/test`` and uses the current repository's input devices, robot models,
qpos mappings and config layout.

Examples:
    # Replay the bundled AVP recording (pickle must be explicitly trusted)
    python example/test/tuning_tool.py --play data/avp1.pkl --trust-pkl \
        --hand left --robot wuji_hand

    # Tune MediaPipe/video parameters
    python example/test/tuning_tool.py --video data/right.mp4 --hand right \
        --robot gaia_hand20 --show-video

    # Tune with a live camera or RealSense
    python example/test/tuning_tool.py --input camera --hand right
    python example/test/tuning_tool.py --input realsense --hand right

Skeleton colors:
    Orange = transformed input keypoints
    Cyan   = target keypoints after scaling parameters
    White  = robot forward-kinematics keypoints
"""

from __future__ import annotations

import argparse
import logging
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Any

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT, TEST_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from viz import TuningViewer

LOGGER = logging.getLogger(__name__)

ROBOT_CHOICES = [
    "shadow_hand",
    "wuji_hand",
    "gaia_hand20",
    "allegro_hand",
    "inspire_hand",
    "ability_hand",
    "leap_hand",
    "svh_hand",
    "linkerhand_l21",
    "linker_l20",
    "rohand",
    "unitree_dex5_hand",
    "sharpa_hand",
]


def resolve_example_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = EXAMPLE_ROOT / path
    return path.resolve()


def _resolve_config(args: argparse.Namespace) -> Path:
    if args.config:
        config_path = resolve_example_path(args.config)
    else:
        profile = args.profile
        if profile == "auto":
            if args.play:
                profile = "avp"
            elif args.input in ("avp", "noitom", "quest3", "pico4"):
                profile = args.input
            else:
                profile = "mediapipe"
        config_path = EXAMPLE_ROOT / "config" / args.optimizer / profile / f"{profile}_{args.robot}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Retarget config not found: {config_path}")
    return config_path.resolve()


def _resolve_viz_config(args: argparse.Namespace) -> Path | None:
    if args.viz_config:
        path = Path(args.viz_config).expanduser()
        if not path.is_absolute():
            # Prefer paths relative to example/, then example/test/ for the copied default.
            example_candidate = EXAMPLE_ROOT / path
            test_candidate = TEST_ROOT / path
            path = example_candidate if example_candidate.exists() else test_candidate
        path = path.resolve()
    else:
        path = TEST_ROOT / "tuning_viz.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Visualization config not found: {path}")
    return path


def _load_trusted_recording(path: Path, trust_pkl: bool) -> list[dict]:
    if not trust_pkl:
        raise ValueError(
            "Refusing to load pickle without explicit trust. "
            "Pass --trust-pkl only for a file you fully trust."
        )
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Expected a list of frames in {path}, got {type(data).__name__}")
    return list(data)


def _create_input_device(args: argparse.Namespace) -> Any:
    if args.input == "camera":
        from input.camera import Camera

        return Camera(camera_id=args.camera_id, show_preview=args.show_video)
    if args.input == "video":
        from input.video import Video

        source = resolve_example_path(args.video or args.source or "data/right.mp4")
        return Video(
            video_path=str(source),
            hand_side=args.hand,
            show_video=args.show_video,
            playback_speed=args.speed,
            loop=True,
            depth_scale=args.depth_scale,
        )
    if args.input == "realsense":
        from input.realsense import Realsense

        return Realsense(hand_side=args.hand, show_video=args.show_video)
    if args.input == "replay":
        if not args.source:
            raise ValueError("--source is required with --input replay")
        if not args.trust_pkl:
            raise ValueError("--input replay loads pickle data; pass --trust-pkl after verifying it")
        from input.mediapipe_replay import MediaPipeReplay

        return MediaPipeReplay(
            record_path=str(resolve_example_path(args.source)),
            playback_speed=args.speed,
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
    raise ValueError(f"Unsupported input device: {args.input}")


def _close_input_device(device: Any) -> None:
    for method_name in ("cleanup", "release", "stop", "close"):
        method = getattr(device, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                LOGGER.warning("Input cleanup via %s() failed: %s", method_name, exc)
            break


def _create_viewer(args: argparse.Namespace) -> TuningViewer:
    config_path = _resolve_config(args)
    viz_config_path = _resolve_viz_config(args)
    print(f"Retarget config: {config_path}")
    print(f"Visualization config: {viz_config_path}")
    return TuningViewer(
        hand_side=args.hand,
        retarget_config_path=str(config_path),
        viz_config_path=str(viz_config_path),
    )


def run_recording_mode(args: argparse.Namespace) -> None:
    recording_path = resolve_example_path(args.play)
    data = _load_trusted_recording(recording_path, args.trust_pkl)
    if args.frames is not None:
        data = data[: args.frames]
    print(f"Loaded {len(data)} frames from {recording_path}")
    viewer = _create_viewer(args)
    viewer.play_recording(data, fps=args.fps)


def run_live_mode(args: argparse.Namespace) -> None:
    input_device = _create_input_device(args)
    viewer = _create_viewer(args)
    hand_key = f"{args.hand}_fingers"
    running = True
    last_result = None

    def stop(_signal, _frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, stop)
    print("=" * 60)
    print(f"Tuning Viewer - {args.input} live input")
    print(f"Hand: {args.hand}")
    print("Edit the retarget YAML while this window is open to hot-reload it.")
    print("=" * 60)

    try:
        with mujoco.viewer.launch_passive(viewer.model, viewer.data) as mj_viewer:
            viewer._set_camera(mj_viewer)
            while mj_viewer.is_running() and running:
                loop_start = time.perf_counter()
                viewer.check_config_reload()
                try:
                    frame = input_device.get_fingers_data()
                    raw = frame.get(hand_key)
                    if raw is not None and np.asarray(raw).shape == (21, 3) and not np.allclose(raw, 0):
                        last_result = viewer._process_frame(np.asarray(raw))
                        viewer.apply_result(last_result)
                except Exception:
                    LOGGER.exception("Failed to process a live tuning frame")

                if last_result is not None:
                    with mj_viewer.lock():
                        viewer.draw_result(mj_viewer.user_scn, last_result)
                mj_viewer.sync()

                elapsed = time.perf_counter() - loop_start
                period = 1.0 / args.fps
                if elapsed < period:
                    time.sleep(period - elapsed)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        _close_input_device(input_device)
    print("Viewer closed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AnyDexRetarget parameter tuning visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", help="Explicit YAML path, absolute or relative to example/")
    parser.add_argument("--viz-config", help="Visualization YAML (default: example/test/tuning_viz.yaml)")
    parser.add_argument("--hand", default="left", choices=["left", "right"])
    parser.add_argument("--robot", default="wuji_hand", choices=ROBOT_CHOICES)
    parser.add_argument("--optimizer", default="adaptive", choices=["adaptive", "vector"])
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "mediapipe", "avp", "noitom", "quest3", "pico4"],
        help="Config input profile used when --config is omitted",
    )
    parser.add_argument("--play", metavar="FILE", help="Play a frame-list .pkl relative to example/")
    parser.add_argument("--trust-pkl", action="store_true", help="Confirm that pickle input is trusted")
    parser.add_argument("--frames", type=int, help="Limit recording playback to the first N frames")
    parser.add_argument(
        "--input",
        choices=["camera", "video", "realsense", "replay", "noitom", "quest3", "avp", "pico4"],
        help="Live/input-device mode",
    )
    parser.add_argument("--video", metavar="FILE", help="Compatibility shorthand for --input video --source FILE")
    parser.add_argument("--realsense", action="store_true", help="Compatibility shorthand for --input realsense")
    parser.add_argument("--source", help="Video/replay source path relative to example/")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--noitom-local-ip", default="192.168.5.25")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", default="udp", choices=["udp", "tcp"])
    parser.add_argument("--avp-ip", default="192.168.50.127")
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be > 0")

    if args.video:
        if args.input and args.input != "video":
            parser.error("--video conflicts with a non-video --input")
        args.input = "video"
        args.source = args.video
    if args.realsense:
        if args.input and args.input != "realsense":
            parser.error("--realsense conflicts with a different --input")
        args.input = "realsense"
    if args.play and args.input:
        parser.error("choose either --play or --input, not both")
    if not args.play and not args.input:
        parser.print_help()
        print("\nChoose --play FILE or --input DEVICE.", file=sys.stderr)
        return 2

    if args.play:
        run_recording_mode(args)
    else:
        run_live_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
