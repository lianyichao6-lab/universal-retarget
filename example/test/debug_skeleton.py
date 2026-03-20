"""Debug visualization: compare MediaPipe input, scaled target, and retargeted FK skeletons.

Shows three hand skeletons side-by-side in the MuJoCo viewer:
  - Blue:  Raw MediaPipe skeleton (after coordinate transform, before scaling)
  - Green: Scaled target skeleton (what the optimizer tries to match)
  - Red:   Robot FK skeleton (retargeting result)

Usage:
    python debug_skeleton.py --config config/leap_hand.yaml
    python debug_skeleton.py --config config/leap_hand.yaml --video data/right.mp4
    python debug_skeleton.py --config config/leap_hand.yaml --input camera
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from qsq_retargeting import Retargeter
from qsq_retargeting.mediapipe import apply_mediapipe_transformations
from qsq_retargeting.opt.base import BaseOptimizer
from input_devices.camera import Camera
from input_devices.video import Video
from input_devices.mediapipe_replay import MediaPipeReplay

# Import teleop_sim configs for qpos_mapping
from teleop_sim import ROBOT_HAND_CONFIGS, map_urdf_to_mujoco_menagerie

# MediaPipe hand connections (pairs of landmark indices)
MP_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

M_TO_CM = 100.0


def draw_skeleton(scn, points, connections, color, radius=0.001, offset=None):
    """Draw a hand skeleton using capsule geoms in the MuJoCo scene.

    Args:
        scn: mujoco.MjvScene (viewer.user_scn)
        points: (N, 3) landmark positions in meters
        connections: list of (i, j) index pairs
        color: (4,) RGBA color
        radius: capsule radius
        offset: (3,) offset to shift the entire skeleton
    """
    if offset is not None:
        points = points + offset

    for i, j in connections:
        if i >= len(points) or j >= len(points):
            continue
        p1 = points[i]
        p2 = points[j]

        if scn.ngeom >= scn.maxgeom:
            break

        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, 0, 0],
            pos=(p1 + p2) / 2,
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        # Set capsule endpoints
        mujoco.mjv_connector(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=p1,
            to=p2,
        )
        scn.ngeom += 1

    # Draw spheres at joint positions
    for idx in range(len(points)):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius * 2, 0, 0],
            pos=points[idx],
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        scn.ngeom += 1


def get_robot_fk_skeleton(optimizer, qpos):
    """Get robot FK positions for visualization.

    Returns:
        points: list of (3,) positions
        connections: list of (i, j) pairs
    """
    robot = optimizer.robot
    robot.compute_forward_kinematics(qpos)

    # Get origin position
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pose = robot.get_link_pose(origin_id)
    origin_pos = origin_pose[:3, 3]

    points = [origin_pos.copy()]  # index 0 = origin
    connections = []

    nf = optimizer.num_fingers
    for fi in range(nf):
        # Get link positions: link1, link3, link4, tip
        link_names = []
        if hasattr(optimizer, 'link1_names') and fi < len(optimizer.link1_names):
            link_names.append(optimizer.link1_names[fi])
        if hasattr(optimizer, 'link3_names') and fi < len(optimizer.link3_names):
            link_names.append(optimizer.link3_names[fi])
        if hasattr(optimizer, 'link4_names') and fi < len(optimizer.link4_names):
            link_names.append(optimizer.link4_names[fi])
        if fi < len(optimizer.task_link_names):
            link_names.append(optimizer.task_link_names[fi])

        prev_idx = 0  # origin
        for lname in link_names:
            lid = robot.get_link_index(lname)
            pose = robot.get_link_pose(lid)
            pos = pose[:3, 3]
            cur_idx = len(points)
            points.append(pos.copy())
            connections.append((prev_idx, cur_idx))
            prev_idx = cur_idx

    return np.array(points), connections


def build_scaled_skeleton(mediapipe_kp, optimizer):
    """Build the scaled target skeleton from MediaPipe keypoints.

    Uses the same scaling/segment_scaling as the optimizer to compute
    the target positions the optimizer tries to match.

    Returns:
        points: (21, 3) scaled keypoints in meters (robot frame)
        connections: same as MP_CONNECTIONS
    """
    # Get origin FK position at current robot state
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    # Compute tip target vectors (wrist -> tip, scaled)
    scaling = optimizer.scaling if hasattr(optimizer, 'scaling') else 1.0
    wrist = mediapipe_kp[0]

    # Build scaled keypoints: apply scaling to all vectors from wrist
    scaled_kp = np.zeros_like(mediapipe_kp)
    scaled_kp[0] = origin_pos  # wrist at robot origin

    # For full-hand vectors, use segment_scaling
    seg_scaling = None
    if hasattr(optimizer, 'segment_scaling'):
        seg_scaling = optimizer.segment_scaling

    # Map MP finger indices
    mp_finger_indices = optimizer.mp_finger_indices if hasattr(optimizer, 'mp_finger_indices') else [0, 1, 2, 3, 4]

    # Apply scaling to each landmark
    for i in range(1, 21):
        vec = mediapipe_kp[i] - wrist  # vector from wrist in meters
        scaled_kp[i] = origin_pos + vec * scaling

    return scaled_kp, MP_CONNECTIONS


def build_raw_skeleton(mediapipe_kp, optimizer):
    """Build the raw MediaPipe skeleton (no scaling, just coordinate transform).

    Places it at the robot origin for comparison.
    """
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    wrist = mediapipe_kp[0]
    raw_kp = np.zeros_like(mediapipe_kp)
    raw_kp[0] = origin_pos

    for i in range(1, 21):
        vec = mediapipe_kp[i] - wrist  # meters, no scaling
        raw_kp[i] = origin_pos + vec

    return raw_kp, MP_CONNECTIONS


def main():
    parser = argparse.ArgumentParser(description="Debug skeleton visualization")
    parser.add_argument("--config", default="config/leap_hand.yaml", help="Config YAML path")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="camera", choices=["camera", "video", "replay"])
    parser.add_argument("--video", default="", help="Video file path")
    parser.add_argument("--play", default="", help="Replay pickle path")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    config_file = EXAMPLE_ROOT / args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    robot_type = config.get('robot', {}).get('type', 'shadow_hand')
    hand_cfg = ROBOT_HAND_CONFIGS.get(robot_type, {})
    model_path = hand_cfg["model_path"](args.hand)

    print(f"Robot: {robot_type}")
    print(f"Model: {model_path}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Initialize control
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            r = model.actuator_ctrlrange[i]
            data.ctrl[i] = (r[0] + r[1]) / 2
        else:
            data.ctrl[i] = 0.0

    for _ in range(100):
        mujoco.mj_step(model, data)

    # Initialize retargeter
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer

    # Initialize input device
    if args.input == "video" or args.video:
        video_path = args.video or "data/right.mp4"
        input_device = Video(
            video_path=video_path,
            hand_side=args.hand,
            show_video=args.show_video,
            playback_speed=args.speed,
            loop=True,
        )
        input_type = "video"
    elif args.input == "replay" or args.play:
        input_device = MediaPipeReplay(
            record_path=args.play,
            playback_speed=args.speed,
            loop=True,
        )
        input_type = "replay"
    else:
        input_device = Camera(camera_id=0, show_preview=True)
        input_type = "camera"

    print(f"Input: {input_type}")

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    cam_cfg = config.get('render', {}).get('camera', {})
    viewer.cam.azimuth = cam_cfg.get('azimuth', 135)
    viewer.cam.elevation = cam_cfg.get('elevation', -20)
    viewer.cam.distance = cam_cfg.get('distance', 0.5)
    viewer.cam.lookat[:] = cam_cfg.get('lookat', [0, 0, 0.05])

    # Colors (RGBA float32)
    COLOR_RAW = np.array([0.2, 0.4, 1.0, 0.6], dtype=np.float32)      # Blue: raw input
    COLOR_SCALED = np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32)    # Green: scaled target
    COLOR_FK = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)        # Red: robot FK

    # Offsets to separate skeletons (left-right)
    OFFSET_RAW = np.array([-0.15, 0, 0])     # Raw on the left
    OFFSET_SCALED = np.array([0, 0, 0])       # Scaled at center (overlaps with robot)
    OFFSET_FK = np.array([0, 0, 0])           # FK at center (robot model position)

    # Shared state
    latest_ctrl = np.zeros(model.nu, dtype=np.float32)
    latest_mediapipe_kp = None
    latest_qpos = None
    data_lock = threading.Lock()
    stop_event = threading.Event()

    def input_thread_fn():
        nonlocal latest_ctrl, latest_mediapipe_kp, latest_qpos
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break

            raw_kp = fingers_data[f"{args.hand}_fingers"]
            if np.allclose(raw_kp, 0):
                time.sleep(0.005)
                continue

            # Get transformed keypoints (same as retargeter internals)
            mediapipe_kp = apply_mediapipe_transformations(raw_kp, args.hand)
            if retargeter.rotation_xyz:
                mediapipe_kp = retargeter._apply_rotation(mediapipe_kp)

            # Retarget
            qpos = retargeter.optimizer.solve(mediapipe_kp)

            # Map to MuJoCo ctrl
            if hand_cfg.get("needs_menagerie_mapping"):
                ctrl = map_urdf_to_mujoco_menagerie(qpos)
            elif "qpos_mapping" in hand_cfg:
                ctrl = qpos[hand_cfg["qpos_mapping"]]
            else:
                ctrl = qpos

            with data_lock:
                if len(ctrl) == model.nu:
                    latest_ctrl[:] = ctrl
                else:
                    min_len = min(len(ctrl), model.nu)
                    latest_ctrl[:min_len] = ctrl[:min_len]
                latest_mediapipe_kp = mediapipe_kp.copy()
                latest_qpos = qpos.copy()

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    print("=" * 60)
    print("Debug Skeleton Viewer")
    print("  Blue  = Raw MediaPipe (no scaling)")
    print("  Green = Scaled target (what optimizer matches)")
    print("  Red   = Robot FK (retargeting result)")
    print("=" * 60)

    try:
        input_thread.start()

        while viewer.is_running():
            with data_lock:
                data.ctrl[:] = latest_ctrl
                mp_kp = latest_mediapipe_kp.copy() if latest_mediapipe_kp is not None else None
                qpos_copy = latest_qpos.copy() if latest_qpos is not None else None

            # Step physics
            for _ in range(10):
                mujoco.mj_step(model, data)

            # Draw debug skeletons
            viewer.user_scn.ngeom = 0  # clear previous frame

            if mp_kp is not None and qpos_copy is not None:
                # 1. Raw MediaPipe skeleton (blue) - offset to left
                raw_pts, raw_conns = build_raw_skeleton(mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, raw_pts, raw_conns, COLOR_RAW,
                              radius=0.0015, offset=OFFSET_RAW)

                # 2. Scaled target skeleton (green) - at robot position
                scaled_pts, scaled_conns = build_scaled_skeleton(mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, scaled_pts, scaled_conns, COLOR_SCALED,
                              radius=0.0015, offset=OFFSET_SCALED)

                # 3. Robot FK skeleton (red) - at robot position
                fk_pts, fk_conns = get_robot_fk_skeleton(optimizer, qpos_copy)
                draw_skeleton(viewer.user_scn, fk_pts, fk_conns, COLOR_FK,
                              radius=0.002, offset=OFFSET_FK)

            viewer.sync()
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        viewer.close()
        print("Done.")


if __name__ == "__main__":
    main()
