"""Debug visualization: compare MediaPipe input, scaled target, and retargeted FK skeletons.
比较初始输入数据，经过scaling之后的，跟经过retargeted之后的骨架

Shows three hand skeletons side-by-side in the MuJoCo viewer:
  - Blue:  Raw MediaPipe skeleton (after coordinate transform, before scaling)
  - Green: Scaled input skeleton immediately before retargeting
  - Red:   Robot FK skeleton (retargeting result)

Usage:
    python debug_skeleton.py --robot leap
    python debug_skeleton.py --robot leap --video data/right.mp4
    python debug_skeleton.py --robot leap --input camera
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

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations
from anydexretarget.optimizer.base_optimizer import BaseOptimizer
from input.camera import Camera
from input.noitom import NoitomInput
from input.video import Video
from input.mediapipe_replay import MediaPipeReplay
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    validate_mujoco_actuator_mapping,
)

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
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue

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
        if not np.isfinite(points[idx]).all():
            continue
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

    def point_pos(link_name, offset=None):
        lid = robot.get_link_index(link_name)
        pose = robot.get_link_pose(lid)
        pos = pose[:3, 3]
        if offset is not None:
            pos = pos + pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
        return pos

    # Get origin position
    origin_pos = point_pos(optimizer.origin_link_name)

    points = [origin_pos.copy()]  # index 0 = origin
    connections = []

    nf = optimizer.num_fingers
    for fi in range(nf):
        # Get link positions: link1, link3, link4, tip
        link_names = []
        if hasattr(optimizer, 'link1_names') and fi < len(optimizer.link1_names):
            link_names.append((optimizer.link1_names[fi], None))
        if hasattr(optimizer, 'link3_names') and fi < len(optimizer.link3_names):
            offset = optimizer.link3_offsets[fi] if hasattr(optimizer, 'link3_offsets') else None
            link_names.append((optimizer.link3_names[fi], offset))
        if hasattr(optimizer, 'link4_names') and fi < len(optimizer.link4_names):
            offset = optimizer.link4_offsets[fi] if hasattr(optimizer, 'link4_offsets') else None
            link_names.append((optimizer.link4_names[fi], offset))
        if fi < len(optimizer.task_link_names):
            offset = optimizer.task_offsets[fi] if hasattr(optimizer, 'task_offsets') else None
            link_names.append((optimizer.task_link_names[fi], offset))

        prev_idx = 0  # origin
        for lname, offset in link_names:
            pos = point_pos(lname, offset)
            cur_idx = len(points)
            points.append(pos.copy())
            connections.append((prev_idx, cur_idx))
            prev_idx = cur_idx

    return np.array(points), connections


def build_scaled_skeleton(mediapipe_kp, optimizer):
    """Build the scaled target skeleton from MediaPipe keypoints.

    Uses the configured scaling/segment_scaling to show the transformed
    human input immediately before it is passed into retargeting.

    Returns:
        points: (21, 3) scaled keypoints in meters (robot frame)
        connections: same as MP_CONNECTIONS
    """
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    scaling = optimizer.scaling if hasattr(optimizer, 'scaling') else 1.0
    wrist = mediapipe_kp[0]

    scaled_kp = np.zeros_like(mediapipe_kp)
    scaled_kp[0] = origin_pos

    mp_finger_indices = optimizer.mp_finger_indices if hasattr(optimizer, 'mp_finger_indices') else [0, 1, 2, 3, 4]

    if hasattr(optimizer, 'segment_scaling_full'):
        seg_full = optimizer.segment_scaling_full
        MP_MCP = [1, 5, 9, 13, 17]
        MP_PIP = [2, 6, 10, 14, 18]
        MP_DIP = [3, 7, 11, 15, 19]
        MP_TIP = [4, 8, 12, 16, 20]
        for local_fi, fi in enumerate(mp_finger_indices):
            # Match AdaptiveOptimizerAnalytical._compute_full_hand_vectors:
            # grow every downstream point from the preceding scaled point.
            mcp_idx, pip_idx, dip_idx, tip_idx = (
                MP_MCP[fi], MP_PIP[fi], MP_DIP[fi], MP_TIP[fi]
            )
            palm = mediapipe_kp[mcp_idx] - wrist
            lateral_scaling = getattr(optimizer, "lateral_scaling", 1.0)
            lateral_axis = getattr(optimizer, "lateral_axis", 1)
            if lateral_scaling != 1.0:
                palm = palm.copy()
                palm[lateral_axis] *= lateral_scaling
            scaled_kp[mcp_idx] = origin_pos + palm * seg_full[local_fi, 0]
            scaled_kp[pip_idx] = scaled_kp[mcp_idx] + (
                mediapipe_kp[pip_idx] - mediapipe_kp[mcp_idx]
            ) * seg_full[local_fi, 1]
            scaled_kp[dip_idx] = scaled_kp[pip_idx] + (
                mediapipe_kp[dip_idx] - mediapipe_kp[pip_idx]
            ) * seg_full[local_fi, 2]
            scaled_kp[tip_idx] = scaled_kp[dip_idx] + (
                mediapipe_kp[tip_idx] - mediapipe_kp[dip_idx]
            ) * seg_full[local_fi, 3]

        # ``mediapipe_kp`` is already the exact robot-specific preprocessed
        # optimizer input.  Do not apply Linker L20 preprocessing a second time
        # here, otherwise the green visualization differs from the real target.
    elif hasattr(optimizer, '_task_kp_indices'):
        for i in range(1, 21):
            scaled_kp[i] = origin_pos + (mediapipe_kp[i] - wrist)
        for origin_kp, task_kp, scale in zip(
            optimizer._origin_kp_indices,
            optimizer._task_kp_indices,
            optimizer._vector_scalings,
        ):
            if origin_kp == 0:
                scaled_kp[task_kp] = origin_pos + (mediapipe_kp[task_kp] - wrist) * scale
            else:
                scaled_kp[task_kp] = scaled_kp[origin_kp] + (mediapipe_kp[task_kp] - mediapipe_kp[origin_kp]) * scale
    else:
        for i in range(1, 21):
            scaled_kp[i] = origin_pos + (mediapipe_kp[i] - wrist) * scaling

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
    parser.add_argument("--config", default=None, help="Config YAML path (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", default="adaptive",
                        choices=["adaptive", "vector"],
                        help="Optimizer type: adaptive (default) or vector (KeyVectorOptimizer)")
    parser.add_argument("--robot", default="leap",
        choices=["shadow", "wuji", "allegro", "leap",
                 "inspire", "ability", "svh", "rohand",
                 "linkerhand_l21", "linker_l20", "l25", "unitree_dex5", "sharpa", "gaia"],
                        help="Robot hand type (default: leap)")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="camera", choices=["camera", "video", "replay", "noitom", "realsense", "avp", "quest3", "pico4"])
    parser.add_argument("--video", default="", help="Video file path")
    parser.add_argument("--play", default="", help="Replay pickle path")
    parser.add_argument("--noitom-local-ip", type=str, default="192.168.5.25")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", type=str, default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    parser.add_argument("--avp-ip", type=str, default="192.168.50.127")
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", type=str, default="udp", choices=["udp", "tcp"])
    parser.add_argument("--pico4-mode", type=str, default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-relay-host", type=str, default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Robot hand transparency (0=invisible, 1=opaque, default: 0.25)")
    args = parser.parse_args()

    robot_name_map = {
        "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
        "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
        "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
        "linker_l20": "linker_l20", "l25": "linkerhand_l25", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand",
        "gaia": "gaia_hand20",
    }
    robot_file = robot_name_map.get(args.robot, args.robot)
    input_to_dir = {"noitom": "noitom", "avp": "avp", "quest3": "quest3", "pico4": "pico4"}
    config_dir = input_to_dir.get(args.input, "mediapipe")
    config_path = args.config if args.config else f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path
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
    actuator_joint_names = validate_mujoco_actuator_mapping(model, hand_cfg)
    if actuator_joint_names:
        print("MuJoCo actuator joint order verified:")
        for actuator_id, joint_name in enumerate(actuator_joint_names):
            print(f"  ctrl[{actuator_id:02d}] -> {joint_name}")

    # Determine control mode (same logic as teleop_sim.py)
    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    direct_qpos_mode = hand_cfg.get("direct_qpos", False)
    qpos_servo_mode = (qpos_servo_alpha is not None) and not direct_qpos_mode
    actuator_mode = (model.nu > 0) and not direct_qpos_mode and not qpos_servo_mode
    if not actuator_mode and not qpos_servo_mode:
        direct_qpos_mode = True
    target_len = model.nu if actuator_mode else model.nq

    # Initialize control
    if actuator_mode:
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                r = model.actuator_ctrlrange[i]
                data.ctrl[i] = (r[0] + r[1]) / 2
            else:
                data.ctrl[i] = 0.0
        for _ in range(100):
            mujoco.mj_step(model, data)
    else:
        mujoco.mj_forward(model, data)

    # Initialize retargeter
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer

    # Initialize input device
    if args.input == "noitom":
        input_device = NoitomInput(
            local_ip=args.noitom_local_ip,
            local_port=args.noitom_local_port,
            server_ip=args.noitom_server_ip,
            server_port=args.noitom_server_port,
        )
        input_type = "noitom"
    elif args.input == "realsense":
        from input.realsense import Realsense
        input_device = Realsense(hand_side=args.hand, show_video=args.show_video)
        input_type = "realsense"
    elif args.input == "avp":
        from input.visionpro import VisionPro
        input_device = VisionPro(ip=args.avp_ip)
        input_type = "avp"
    elif args.input == "quest3":
        from input.quest3 import Quest3
        input_device = Quest3(port=args.quest3_port, protocol=args.quest3_protocol)
        input_type = "quest3"
    elif args.input == "pico4":
        from input.pico4 import Pico4
        input_device = Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
        input_type = "pico4"
    elif args.input == "video" or args.video:
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
    if input_type == "pico4":
        if args.pico4_mode == "direct":
            print(f"Pico4 mode: direct (tcp={args.pico4_port}, udp_broadcast={args.pico4_broadcast_port})")
        else:
            print(f"Pico4 mode: relay ({args.pico4_relay_host}:{args.pico4_relay_port})")
    if qpos_servo_mode:
        print(f"Control mode: qpos servo (alpha={qpos_servo_alpha})")
    elif actuator_mode:
        print("Control mode: actuator position")
    else:
        print("Control mode: direct qpos")

    # Make only the hand semi-transparent; keep the shared scene floor opaque.
    hand_geom_mask = model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_rgba[hand_geom_mask, 3] = args.alpha

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    cam_cfg = config.get('render', {}).get('camera', {})
    viewer.cam.azimuth = cam_cfg.get('azimuth', 135)
    viewer.cam.elevation = cam_cfg.get('elevation', -20)
    viewer.cam.distance = cam_cfg.get('distance', 0.5)
    viewer.cam.lookat[:] = cam_cfg.get('lookat', [0, 0, 0.05])

    # Colors (RGBA float32)
    COLOR_RAW = np.array([0.2, 0.4, 1.0, 0.6], dtype=np.float32)      # Blue: raw input
    COLOR_SCALED = np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32)    # Green: scaled input
    COLOR_FK = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)        # Red: robot FK

    # Offsets to separate skeletons (left-right)
    OFFSET_RAW = np.array([-0.15, 0, 0])     # Raw on the left
    OFFSET_SCALED = np.array([0, 0, 0])       # Scaled at center (overlaps with robot)
    OFFSET_FK = np.array([0, 0, 0])           # FK at center (robot model position)

    # Shared state
    latest_target = np.zeros(target_len, dtype=np.float32)
    latest_raw_mediapipe_kp = None
    latest_retarget_input_kp = None
    latest_qpos = None
    data_lock = threading.Lock()
    stop_event = threading.Event()

    def input_thread_fn():
        nonlocal latest_raw_mediapipe_kp, latest_retarget_input_kp, latest_qpos
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break

            raw_kp = fingers_data[f"{args.hand}_fingers"]
            if np.allclose(raw_kp, 0):
                time.sleep(0.005)
                continue

            # Raw blue skeleton: transformed/rotated Pico input before
            # robot-specific preprocessing.
            raw_mediapipe_kp = apply_mediapipe_transformations(raw_kp, args.hand)
            if retargeter.rotation_xyz:
                raw_mediapipe_kp = retargeter._apply_rotation(raw_mediapipe_kp)

            # Use exactly the same end-to-end path as teleop_sim.py.  The
            # verbose keypoints are the keypoints actually consumed by
            # optimizer.solve; the returned qpos also includes the configured
            # low-pass filter, just like normal teleoperation.
            qpos, verbose = retargeter.retarget_verbose(raw_kp, apply_filter=True)
            retarget_input_kp = verbose["mediapipe_kp"]

            # Map the full retarget output to the exact MuJoCo actuator/qpos
            # order. GEORT L20 is mapped by independent joint names.
            target = map_retarget_qpos(
                qpos,
                hand_cfg,
                retargeter.optimizer.robot.dof_joint_names,
            )
            target = np.asarray(target, dtype=np.float32)

            with data_lock:
                n = min(len(target), target_len)
                latest_target[:n] = target[:n]
                latest_raw_mediapipe_kp = raw_mediapipe_kp.copy()
                latest_retarget_input_kp = retarget_input_kp.copy()
                latest_qpos = qpos.copy()

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    # Align the Pinocchio/URDF frame to the matching MuJoCo body frame.
    # This includes translation: the GEORT scene places hand_base_link at
    # z=0.05 while the URDF root is at z=0.
    pin_origin_id = optimizer.robot.get_link_index(optimizer.origin_link_name)
    pin_origin_pose = optimizer.robot.get_link_pose(pin_origin_id)
    pin_origin_pos = pin_origin_pose[:3, 3].copy()
    pin_origin_rot = pin_origin_pose[:3, :3].copy()
    mujoco_origin_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, optimizer.origin_link_name
    )
    if mujoco_origin_body_id < 0:
        # Fallback for a hand-only scene whose root body has a different name.
        root_body_ids = [
            body_id
            for body_id in range(1, model.nbody)
            if model.body_parentid[body_id] == 0
        ]
        if len(root_body_ids) == 1:
            mujoco_origin_body_id = root_body_ids[0]

    frame_rot_matrix = None
    frame_origin_pos = None
    if mujoco_origin_body_id >= 0:
        mujoco.mj_forward(model, data)
        frame_origin_pos = data.xpos[mujoco_origin_body_id].copy()
        mujoco_origin_rot = data.xmat[mujoco_origin_body_id].reshape(3, 3).copy()
        frame_rot_matrix = mujoco_origin_rot @ pin_origin_rot.T

    # Legacy fallback for models that expose only a fixed base rotation.
    base_quat = hand_cfg.get("base_quat")
    if frame_rot_matrix is None and base_quat is not None:
        from scipy.spatial.transform import Rotation
        # MuJoCo quat is (w, x, y, z), scipy expects (x, y, z, w)
        base_rot = Rotation.from_quat(
            [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
        )
        base_rot_matrix = base_rot.as_matrix()
    else:
        base_rot_matrix = None

    def align_points_to_mujoco(pts):
        """Transform URDF-frame skeleton points into the MuJoCo world frame."""
        if frame_rot_matrix is not None:
            return (pts - pin_origin_pos) @ frame_rot_matrix.T + frame_origin_pos
        if base_rot_matrix is not None:
            return pts @ base_rot_matrix.T
        return pts

    print("=" * 60)
    print("Debug Skeleton Viewer")
    print("  Blue  = Raw MediaPipe (no scaling)")
    print("  Green = Scaled input before retargeting")
    print("  Red   = Robot FK (retargeting result)")
    print("=" * 60)

    try:
        input_thread.start()

        while viewer.is_running():
            with data_lock:
                target_copy = latest_target.copy()
                raw_mp_kp = (latest_raw_mediapipe_kp.copy()
                              if latest_raw_mediapipe_kp is not None else None)
                input_mp_kp = (latest_retarget_input_kp.copy()
                                if latest_retarget_input_kp is not None else None)
                qpos_copy = latest_qpos.copy() if latest_qpos is not None else None

            # Apply control
            if direct_qpos_mode:
                data.qpos[:target_len] = target_copy
                mujoco.mj_forward(model, data)
            elif qpos_servo_mode:
                data.qpos[:target_len] += float(qpos_servo_alpha) * (target_copy - data.qpos[:target_len])
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            elif actuator_mode:
                data.ctrl[:] = target_copy
                for _ in range(10):
                    mujoco.mj_step(model, data)

            # Draw debug skeletons
            viewer.user_scn.ngeom = 0  # clear previous frame

            if raw_mp_kp is not None and input_mp_kp is not None and qpos_copy is not None:
                # 1. Raw MediaPipe skeleton (blue) - offset to left
                raw_pts, raw_conns = build_raw_skeleton(raw_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(raw_pts), raw_conns, COLOR_RAW,
                              radius=0.0015, offset=OFFSET_RAW)

                # 2. Scaled input skeleton (green) - before retargeting
                scaled_pts, scaled_conns = build_scaled_skeleton(input_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(scaled_pts), scaled_conns, COLOR_SCALED,
                              radius=0.0015, offset=OFFSET_SCALED)

                # 3. Robot FK skeleton (red) - retargeting output
                fk_pts, fk_conns = get_robot_fk_skeleton(optimizer, qpos_copy)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(fk_pts), fk_conns, COLOR_FK,
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
