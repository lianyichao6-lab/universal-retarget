#!/usr/bin/env python3
"""MuJoCo comparison viewer for a HUG prediction or canonical grasp state."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'example'
for item in (ROOT, EXAMPLE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
from anydexretarget import Retargeter
from anydexretarget.hug_adapter import load_prediction
from anydexretarget.hand_representation import BONE_EDGES, load_canonical_grasp_state
from anydexretarget.l25_target_chain import (
    build_l25_target_chain,
    chain_error_metrics,
    l25_chain_points,
)
from output.sim.mujoco_output import ROBOT_HAND_CONFIGS, map_retarget_qpos
from test.debug_skeleton import draw_skeleton, build_raw_skeleton, build_scaled_skeleton, get_robot_fk_skeleton

CONFIGS = {
    'vector': EXAMPLE / 'config/vector/mediapipe/mediapipe_linkerhand_l25.yaml',
    'adaptive': EXAMPLE / 'config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml',
}
COLORS = {
    'raw': np.array([0.2, 0.4, 1.0, 0.75], dtype=np.float32),
    'scaled': np.array([0.2, 0.9, 0.3, 0.75], dtype=np.float32),
    'fk': np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32),
}


def build_exact_vector_targets(optimizer, transformed_keypoints):
    """Visualize the actual Vector objective without inventing missing joints."""
    if not hasattr(optimizer, '_compute_target_vectors'):
        raise TypeError('Exact Vector target display requires KeyVectorOptimizer')
    points = np.full((21, 3), np.nan, dtype=np.float64)
    target_vectors_m = optimizer._compute_target_vectors(transformed_keypoints) / 100.0
    for origin_kp, task_kp, origin_link, vector in zip(
        optimizer._origin_kp_indices,
        optimizer._task_kp_indices,
        optimizer._kv_origin_indices,
        target_vectors_m,
    ):
        origin = optimizer.robot.get_link_pose(int(origin_link))[:3, 3]
        points[origin_kp] = origin
        points[task_kp] = origin + vector
    connections = [
        (0, 2), (2, 3), (3, 4),
        (0, 6), (6, 7), (7, 8),
        (0, 10), (10, 11), (11, 12),
        (0, 14), (14, 15), (15, 16),
        (0, 18), (18, 19), (19, 20),
    ]
    return points, connections

def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose MuJoCo wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prediction', type=Path, help='Official HUG grasp_pred/*.pkl file.')
    source.add_argument('--canonical', type=Path, help='CanonicalGraspState .npz file.')
    parser.add_argument('--robot', choices=['l25'], default='l25')
    parser.add_argument('--optimizer', choices=['vector', 'adaptive'], default='vector')
    parser.add_argument(
        '--green-mode',
        choices=['vector', 'continuous'],
        default='vector',
        help='Vector shows exact current objectives; continuous shows MANO directions with L25 FK segment lengths.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Optional native L25 YAML. Overrides the default selected by --optimizer.',
    )
    parser.add_argument('--hand', choices=['right'], default='right')
    parser.add_argument('--alpha', type=float, default=0.28)
    parser.add_argument('--camera-azimuth', type=float, default=135.0, help='MuJoCo viewing azimuth in degrees; does not affect retargeting.')
    parser.add_argument('--base-yaw-deg', type=float, default=0.0, help='Rotate the complete L25 and skeleton overlay about hand_base_link local Z; does not affect qpos.')
    args = parser.parse_args()
    canonical_state = None
    if args.canonical is not None:
        canonical_state = load_canonical_grasp_state(args.canonical)
        keypoints_camera = canonical_state.keypoints_for_retargeting()
        input_label = f'canonical={args.canonical}'
    else:
        frame = load_prediction(args.prediction)
        keypoints_camera = frame.keypoints_3d
        input_label = f'prediction={args.prediction}'
    config_path = args.config if args.config is not None else CONFIGS[args.optimizer]
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    retargeter = Retargeter.from_yaml(str(config_path), hand_side=args.hand)
    optimizer = retargeter.optimizer
    qpos, verbose = retargeter.retarget_verbose(keypoints_camera, apply_filter=False)
    if canonical_state is not None:
        # Compare in the original camera frame. verbose['mediapipe_kp'] is
        # already transformed and optionally rotated by the retargeter.
        canonical_error = float(np.max(np.abs(
            keypoints_camera - canonical_state.keypoints_for_retargeting()
        )))
    hand_cfg = ROBOT_HAND_CONFIGS['linkerhand_l25']
    target = map_retarget_qpos(qpos, hand_cfg, optimizer.robot.dof_joint_names)
    model_path = hand_cfg['model_path'](args.hand)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, optimizer.origin_link_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {optimizer.origin_link_name}")
    yaw = np.deg2rad(args.base_yaw_deg)
    yaw_quat = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    model.body_quat[body_id] = _quat_multiply(model.body_quat[body_id].copy(), yaw_quat)
    data = mujoco.MjData(model)
    data.qpos[:len(target)] = target
    mujoco.mj_forward(model, data)
    mask = model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_rgba[mask, 3] = args.alpha
    pin_origin_id = optimizer.robot.get_link_index(optimizer.origin_link_name)
    pin_pose = optimizer.robot.get_link_pose(pin_origin_id)
    pin_origin_pos = pin_pose[:3, 3].copy()
    pin_origin_rot = pin_pose[:3, :3].copy()
    mujoco.mj_forward(model, data)
    frame_origin_pos = data.xpos[body_id].copy()
    frame_rot = data.xmat[body_id].reshape(3, 3) @ pin_origin_rot.T
    def align(points):
        return (points - pin_origin_pos) @ frame_rot.T + frame_origin_pos
    raw_pts, conns = build_raw_skeleton(verbose['mediapipe_kp'], optimizer)
    if args.green_mode == 'continuous':
        target_chain = build_l25_target_chain(optimizer, verbose['mediapipe_kp'])
        scaled_pts = target_chain.target_points
        scaled_conns = [tuple(edge) for edge in BONE_EDGES]
        green_label = 'Green=continuous MANO-direction target with L25 FK segment lengths.'
    elif hasattr(optimizer, '_compute_target_vectors'):
        scaled_pts, scaled_conns = build_exact_vector_targets(
            optimizer, verbose['mediapipe_kp']
        )
        green_label = 'Green=exact Vector objective points (not an inferred 21-joint skeleton).'
    else:
        scaled_pts, scaled_conns = build_scaled_skeleton(verbose['mediapipe_kp'], optimizer)
        green_label = 'Green=scaled optimizer target skeleton.'
    fk_pts = l25_chain_points(optimizer, qpos)
    fk_conns = [tuple(edge) for edge in BONE_EDGES]
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0.02, 0, 0.08]
    print(input_label)
    print('Blue=retarget-frame human skeleton, Red=L25 FK.')
    print(green_label)
    if args.green_mode == 'continuous':
        metrics = chain_error_metrics(target_chain, fk_pts)
        print(
            'Continuous-target diagnostics: '
            f"mean_point={metrics['mean_point_error_cm']:.2f} cm, "
            f"tip_mean={metrics['tip_mean_error_cm']:.2f} cm, "
            f"segment_direction={metrics['mean_segment_direction_error_deg']:.1f} deg, "
            f"thumb_index={metrics['thumb_index_distance_error_cm']:.2f} cm"
        )
    print(f'Base yaw: {args.base_yaw_deg:.1f} deg (visual placement only; qpos unchanged).')
    if canonical_state is not None:
        print(
            'Canonical state: '
            f'21x3={canonical_state.keypoints_canonical.shape}, '
            f'MANO mesh={canonical_state.mano_mesh_vertices_camera.shape}, '
            f'object point={canonical_state.object_point_camera.tolist()}, '
            f'canonical round-trip delta={canonical_error:.3g}'
        )
    print('Close the MuJoCo window to exit.')
    try:
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            draw_skeleton(viewer.user_scn, align(raw_pts), conns, COLORS['raw'], radius=0.0015, offset=frame_rot @ np.array([-0.15, 0, 0]))
            draw_skeleton(viewer.user_scn, align(scaled_pts), scaled_conns, COLORS['scaled'], radius=0.0015)
            draw_skeleton(viewer.user_scn, align(fk_pts), fk_conns, COLORS['fk'], radius=0.002)
            viewer.sync()
            time.sleep(0.02)
    finally:
        viewer.close()

if __name__ == '__main__':
    main()
