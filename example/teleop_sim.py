"""Teleoperation with MuJoCo Simulation.

Uses the Retargeter interface to map hand tracking input to robot hand joint angles,
visualized in MuJoCo simulation. Supports multiple robot hands.

Usage:
    # Shadow Hand (default)
    python teleop_sim.py --play data/avp1.pkl --hand left

    # Specify robot hand type
    python teleop_sim.py --config config/allegro_hand.yaml --input camera --hand right
    python teleop_sim.py --config config/inspire_hand.yaml --play data/avp1.pkl
    python teleop_sim.py --config config/wuji_hand.yaml --input camera --hand right

    # Video file input
    python teleop_sim.py --video data/right.mp4 --hand right
    python teleop_sim.py --video data/right.mp4 --hand right --show-video

    # Live VisionPro input
    python teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

    # Record input data while using VisionPro
    python teleop_sim.py --input visionpro --record

Supported robot hands:
- shadow_hand (default)
- wuji_hand
- allegro_hand
- inspire_hand
- ability_hand
- leap_hand
- svh_hand

Input device types:
- visionpro: Live VisionPro input
- mediapipe_replay: Replay recorded MediaPipe hand tracking data
- camera: Live laptop/USB camera input with MediaPipe
- video: MP4/AVI video file with MediaPipe hand detection
- realsense: Intel RealSense camera with MediaPipe
"""

import argparse
import pickle
import sys
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsq_retargeting import Retargeter
from input_devices.visionpro import VisionPro
from input_devices.mediapipe_replay import MediaPipeReplay
from input_devices.camera import Camera
from input_devices.quest3 import Quest3
from input_devices.realsense import Realsense
from input_devices.video import Video


# Robot hand configurations for MuJoCo visualization
# model_path: MJCF file path for MuJoCo rendering
# qpos_mapping: maps Pinocchio joint order -> MuJoCo actuator order (index list)
# needs_menagerie_mapping: special mapping for Shadow Hand Menagerie (tendon coupling)
ROBOT_HAND_CONFIGS = {
    "shadow_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "shadow_hand" / f"scene_{side}.xml"),
        "needs_menagerie_mapping": True,
    },
    "wuji_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "wuji_hand" / "right.xml"),
    },
    "allegro_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "allegro_hand" / "scene_right.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "inspire_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "inspire_hand" / "inspire_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "ability_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "ability_hand" / "ability_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "leap_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "leap_hand" / "leap_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "svh_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "schunk_hand" / "schunk_svh_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 13, 14, 15, 16, 9, 10, 11, 12, 4, 5, 6, 7, 17, 18, 19],
    },
}


def map_urdf_to_mujoco_menagerie(qpos: np.ndarray) -> np.ndarray:
    """Map URDF joint angles (22 DoF) to MuJoCo Menagerie actuators (20 DoF).

    URDF order (Pinocchio, alphabetically ordered by joint name):
        [0:4]   FFJ4, FFJ3, FFJ2, FFJ1
        [4:9]   LFJ5, LFJ4, LFJ3, LFJ2, LFJ1
        [9:13]  MFJ4, MFJ3, MFJ2, MFJ1
        [13:17] RFJ4, RFJ3, RFJ2, RFJ1
        [17:22] THJ5, THJ4, THJ3, THJ2, THJ1

    MuJoCo order (Menagerie actuators):
        [0:2]   WRJ2, WRJ1 (wrist - not in URDF)
        [2:7]   THJ5, THJ4, THJ3, THJ2, THJ1
        [7:10]  FFJ4, FFJ3, FFJ0 (tendon couples J2+J1)
        [10:13] MFJ4, MFJ3, MFJ0
        [13:16] RFJ4, RFJ3, RFJ0
        [16:20] LFJ5, LFJ4, LFJ3, LFJ0
    """
    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[0] = 0.0  # WRJ2
    ctrl[1] = 0.0  # WRJ1
    ctrl[2] = qpos[17]  # THJ5
    ctrl[3] = qpos[18]  # THJ4
    ctrl[4] = qpos[19]  # THJ3
    ctrl[5] = qpos[20]  # THJ2
    ctrl[6] = qpos[21]  # THJ1
    ctrl[7] = qpos[0]   # FFJ4
    ctrl[8] = qpos[1]   # FFJ3
    ctrl[9] = qpos[2] + qpos[3]  # FFJ0 tendon
    ctrl[10] = qpos[9]   # MFJ4
    ctrl[11] = qpos[10]  # MFJ3
    ctrl[12] = qpos[11] + qpos[12]  # MFJ0 tendon
    ctrl[13] = qpos[13]  # RFJ4
    ctrl[14] = qpos[14]  # RFJ3
    ctrl[15] = qpos[15] + qpos[16]  # RFJ0 tendon
    ctrl[16] = qpos[4]   # LFJ5
    ctrl[17] = qpos[5]   # LFJ4
    ctrl[18] = qpos[6]   # LFJ3
    ctrl[19] = qpos[7] + qpos[8]  # LFJ0 tendon
    return ctrl


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/adaptive_analytical_avp.yaml",
    input_device_type: str = "mediapipe_replay",
    mediapipe_replay_path: str = "",
    video_path: str = "",
    visionpro_ip: str = "192.168.50.127",
    quest3_port: int = 9000,
    quest3_protocol: str = "udp",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    show_video: bool = False,
):
    """Run teleoperation with MuJoCo simulation.

    Input device runs in a background thread to avoid blocking the MuJoCo
    render loop. The main loop steps physics and syncs the viewer at a
    steady rate, picking up the latest control signals from the input thread.
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Load config to determine robot type
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    robot_type = config.get('robot', {}).get('type', 'shadow_hand')

    # Look up robot hand config
    if robot_type not in ROBOT_HAND_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}. "
                         f"Supported: {list(ROBOT_HAND_CONFIGS.keys())}")

    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
    model_path = Path(hand_cfg["model_path"](hand_side))

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    print(f"  Robot: {robot_type}")
    print(f"  Model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Initialize control signals
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0

    # Stabilize model
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Launch viewer with camera from YAML config
    viewer = mujoco.viewer.launch_passive(model, data)
    cam_cfg = config.get('render', {}).get('camera', {})
    viewer.cam.azimuth = cam_cfg.get('azimuth', 135)
    viewer.cam.elevation = cam_cfg.get('elevation', -20)
    viewer.cam.distance = cam_cfg.get('distance', 0.5)
    viewer.cam.lookat[:] = cam_cfg.get('lookat', [0, 0, 0.05])

    # Initialize input device
    device_map = {
        "visionpro": lambda: VisionPro(ip=visionpro_ip),
        "quest3": lambda: Quest3(port=quest3_port, protocol=quest3_protocol),
        "mediapipe_replay": lambda: MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
        "camera": lambda: Camera(camera_id=0, show_preview=True),
        "realsense": lambda: Realsense(
            hand_side=hand_side,
            show_video=show_video,
        ),
        "video": lambda: Video(
            video_path=video_path,
            hand_side=hand_side,
            show_video=show_video,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        raise ValueError("mediapipe_replay_path is required for mediapipe_replay mode")

    input_device = device_map[input_device_type]()

    # Initialize retargeter
    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Disable recording when using replay mode
    if input_device_type in ("mediapipe_replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    # Prepare recording
    input_data_log = [] if enable_recording else None
    start_time = time.time()

    # --- Threaded input: decouple input device from MuJoCo render loop ---
    # Shared state between input thread and main thread
    latest_ctrl = np.zeros(model.nu, dtype=np.float32)
    ctrl_lock = threading.Lock()
    ctrl_ready = False
    stop_event = threading.Event()
    input_frame_count = 0

    def input_thread_fn():
        nonlocal latest_ctrl, ctrl_ready, input_frame_count
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break
            fingers_pose = fingers_data[f"{hand_side}_fingers"]

            if np.allclose(fingers_pose, 0):
                time.sleep(0.005)
                continue

            # Record raw input data if enabled
            if enable_recording and input_data_log is not None:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": fingers_data["left_fingers"].copy(),
                    "right_fingers": fingers_data["right_fingers"].copy(),
                })

            # Retarget to joint angles
            qpos = retargeter.retarget(fingers_pose)

            # Map Pinocchio joint order -> MuJoCo actuator order
            if hand_cfg.get("needs_menagerie_mapping"):
                ctrl = map_urdf_to_mujoco_menagerie(qpos)
            elif "qpos_mapping" in hand_cfg:
                ctrl = qpos[hand_cfg["qpos_mapping"]]
            else:
                ctrl = qpos

            with ctrl_lock:
                if len(ctrl) == model.nu:
                    latest_ctrl[:] = ctrl
                else:
                    min_len = min(len(ctrl), model.nu)
                    latest_ctrl[:min_len] = ctrl[:min_len]
                ctrl_ready = True
            input_frame_count += 1

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print(f"Starting teleoperation...")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        print("=" * 50)

        input_thread.start()

        render_count = 0
        fps_start_time = time.time()
        sim_dt = model.opt.timestep
        n_substeps = 10
        render_interval = sim_dt * n_substeps  # target time per render frame

        while viewer.is_running():
            loop_start = time.time()

            # Pick up latest control from input thread
            with ctrl_lock:
                if ctrl_ready:
                    data.ctrl[:] = latest_ctrl

            # Step simulation
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            viewer.sync()

            # FPS counter
            render_count += 1
            if render_count % 200 == 0:
                elapsed = time.time() - fps_start_time
                render_fps = render_count / elapsed
                print(f"Render FPS: {render_fps:.1f}  |  Input FPS: {input_frame_count / elapsed:.1f}")

            # Sleep to maintain target render rate
            elapsed_this_frame = time.time() - loop_start
            sleep_time = render_interval - elapsed_this_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        viewer.close()

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description='Teleoperation with MuJoCo Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay MediaPipe recording
  python teleop_sim.py --play data/avp1.pkl --hand left

  # Live camera input
  python teleop_sim.py --input camera --hand right

  # Live VisionPro input
  python teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

  # Record input data while using VisionPro
  python teleop_sim.py --input visionpro --record
        """
    )

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (default: auto-select based on input device)')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'],
                        help='Hand side (default: right)')

    # Input device options
    parser.add_argument('--input', type=str, default=None,
                        choices=['visionpro', 'quest3', 'mediapipe_replay', 'camera', 'realsense', 'video'],
                        help='Input device type')
    parser.add_argument('--realsense', action='store_true',
                        help='Use RealSense camera (shortcut for --input realsense)')
    parser.add_argument('--show-video', action='store_true',
                        help='Show video with MediaPipe landmarks overlay')

    # Shortcut options
    parser.add_argument('--play', type=str, default=None, metavar='FILE',
                        help='Play MediaPipe recording file (shortcut for --input mediapipe_replay)')
    parser.add_argument('--video', type=str, default=None, metavar='FILE',
                        help='Play MP4/AVI video file with MediaPipe detection (shortcut for --input video)')

    # VisionPro options
    parser.add_argument('--ip', type=str, default='192.168.50.127',
                        help='VisionPro IP address (default: 192.168.50.127)')

    # Quest 3 options
    parser.add_argument('--port', type=int, default=9000,
                        help='Quest 3 HTS listener port (default: 9000)')
    parser.add_argument('--protocol', type=str, default='udp', choices=['udp', 'tcp'],
                        help='Quest 3 HTS transport protocol (default: udp)')

    # Playback options
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for replay mode (default: 1.0)')
    parser.add_argument('--no-loop', action='store_true',
                        help='Disable looping for replay mode')

    # Recording
    parser.add_argument('--record', action='store_true',
                        help='Record input data to file')
    parser.add_argument('--output', type=str, default=None, metavar='FILE',
                        help='Output file for recording (default: auto-generated)')

    args = parser.parse_args()

    # Determine input device type and paths
    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    # Default to mediapipe_replay with example data if no input specified
    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    # Validate paths
    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")

    # Auto-select config based on input device if not specified
    config_path = args.config
    if config_path is None:
        config_map = {
            "quest3": "config/adaptive_analytical_quest3.yaml",
        }
        config_path = config_map.get(input_device_type, "config/adaptive_analytical_avp.yaml")

    # Run teleoperation
    log = run_teleop(
        hand_side=args.hand,
        config_path=config_path,
        input_device_type=input_device_type,
        mediapipe_replay_path=mediapipe_replay_path,
        video_path=video_path,
        visionpro_ip=args.ip,
        quest3_port=args.port,
        quest3_protocol=args.protocol,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
        show_video=args.show_video,
    )

    # Save recording if enabled
    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"

        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
