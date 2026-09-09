#!/usr/bin/env python3
"""Create a no-scale O30 distal-pad object-relative grasp from HUG contacts."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.o30_retarget_backend import BACKENDS, VECTOR_CONFIG, retarget_o30_static
from anydexretarget.o30_scale import O30ScaleProfile

ROOT = Path(__file__).resolve().parents[1]
O30_MODEL = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3 or source.shape[1] != 3:
        raise ValueError("Rigid fit requires matching N x 3 arrays with N >= 3")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation, target_center - rotation @ source_center


def _apply(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation[None]


def _gaps(value: str | None, names: list[str], default: float) -> np.ndarray:
    result = np.full(len(names), default, dtype=np.float64)
    if not value:
        return result
    index = {name.lower(): position for position, name in enumerate(names)}
    for item in value.split(','):
        name, marker, raw = item.strip().partition('=')
        if not marker or name.lower() not in index:
            raise ValueError("--surface-gaps-mm entries must be finger=millimeters")
        gap = float(raw)
        if gap < 0:
            raise ValueError("--surface-gaps-mm values must be non-negative")
        result[index[name.lower()]] = gap / 1000.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact-plan', type=Path, required=True)
    parser.add_argument('--scale-profile', type=Path, help='Validated O30 Vector scale profile JSON')
    parser.add_argument('--backend', choices=BACKENDS, default='vector')
    parser.add_argument('--config', type=Path, help='Optional native Vector or Adaptive config override.')
    parser.add_argument('--dex-scaling', type=float)
    parser.add_argument('--dex-project-dist', type=float)
    parser.add_argument('--dex-escape-dist', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-fingers', type=str)
    parser.add_argument('--surface-gap-mm', type=float, default=0.0)
    parser.add_argument('--surface-gaps-mm', type=str)
    parser.add_argument('--contact-weight', type=float, default=1.0)
    parser.add_argument('--posture-weight', type=float, default=0.12)
    parser.add_argument('--contact-scale-mm', type=float, default=10.0)
    parser.add_argument('--max-evaluations', type=int, default=180)
    parser.add_argument('--wrist-translation-limit-mm', type=float, default=0.0, help='Allow bounded O30 wrist/object translation compensation.')
    parser.add_argument('--wrist-rotation-limit-deg', type=float, default=0.0, help='Allow bounded O30 wrist/object rotation compensation.')
    parser.add_argument('--wrist-position-weight', type=float, default=0.35)
    parser.add_argument('--wrist-orientation-weight', type=float, default=0.25)
    args = parser.parse_args()
    scale_profile = O30ScaleProfile.load(args.scale_profile) if args.scale_profile else None
    if args.surface_gap_mm < 0 or args.contact_weight <= 0 or args.posture_weight < 0 or args.contact_scale_mm <= 0 or args.max_evaluations <= 0 or args.wrist_translation_limit_mm < 0 or args.wrist_rotation_limit_deg < 0 or args.wrist_position_weight < 0 or args.wrist_orientation_weight < 0:
        raise ValueError('Invalid optimization settings')
    with np.load(args.contact_plan, allow_pickle=False) as data:
        contact = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {'source_canonical_grasp', 'near_surface', 'finger_names', 'fingertip_positions_camera', 'contact_point_positions_camera', 'contact_point_alpha', 'surface_anchor_camera', 'surface_normal_camera'}
    if missing := required - set(contact):
        raise ValueError('Contact plan missing: ' + ', '.join(sorted(missing)))
    active = np.asarray(contact['near_surface'], dtype=np.uint8).astype(bool)
    finger_names = [str(item) for item in contact['finger_names']]
    if args.contact_fingers:
        selected = {item.strip().lower() for item in args.contact_fingers.split(',') if item.strip()}
        unknown = selected - {item.lower() for item in finger_names}
        if unknown:
            raise ValueError('Unknown contact finger(s): ' + ', '.join(sorted(unknown)))
        active = np.asarray([item.lower() in selected for item in finger_names])
    if int(active.sum()) < 3:
        raise ValueError('Need at least three selected O30 contact fingers')
    state_path = Path(str(contact['source_canonical_grasp'].item()))
    state = load_canonical_grasp_state(state_path)
    human_tips = np.asarray(contact['fingertip_positions_camera'], dtype=np.float64)
    human_contacts = np.asarray(contact['contact_point_positions_camera'], dtype=np.float64)
    alpha = np.asarray(contact['contact_point_alpha'], dtype=np.float64)
    if human_tips.shape != (5, 3) or human_contacts.shape != (5, 3) or alpha.shape != (5,) or np.any((alpha <= 0) | (alpha > 1)):
        raise ValueError('Invalid HUG distal-pad contact geometry')

    baseline = retarget_o30_static(
        state.keypoints_for_retargeting(), backend=args.backend, native_config=args.config,
        scale_profile=scale_profile, dex_scaling=args.dex_scaling, dex_project_dist=args.dex_project_dist,
        dex_escape_dist=args.dex_escape_dist,
    )
    retargeter, robot = baseline.geometry_retargeter, baseline.geometry_retargeter.optimizer.robot
    vector_names, baseline_q = baseline.joint_names, np.asarray(baseline.qpos, dtype=np.float64)
    task_names = [str(item) for item in retargeter.optimizer.task_link_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    if len(task_names) != 5 or task_offsets.shape != (5, 3):
        raise ValueError('O30 Vector must expose five audited fingertip task points')
    task_ids = [robot.get_link_index(name) for name in task_names]
    contact_offsets = task_offsets * alpha[:, None]
    model = mujoco.MjModel.from_xml_path(str(O30_MODEL))
    model_names = [str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in range(model.njnt)]
    by_model = {name.lower(): index for index, name in enumerate(model_names)}
    indices = np.asarray([by_model[name.lower()] for name in vector_names], dtype=np.int64)
    lower, upper = model.jnt_range[indices, 0], model.jnt_range[indices, 1]
    baseline_q = np.clip(baseline_q, lower + 1e-6, upper - 1e-6)

    def fk(qpos: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return robot.compute_points_batch(np.asarray(qpos, dtype=np.float64), task_ids, offsets)

    baseline_tips, baseline_contacts = fk(baseline_q, task_offsets), fk(baseline_q, contact_offsets)
    if int(active.sum()) == 2:
        fit_source = np.vstack((human_contacts[active], state.wrist_position_camera))
        fit_target = np.vstack((baseline_contacts[active], np.zeros(3)))
        reference = 'two_contacts_plus_wrist'
    else:
        fit_source, fit_target, reference = human_contacts[active], baseline_contacts[active], 'active_contacts'
    rotation, translation = _rigid(fit_source, fit_target)
    anchors = _apply(np.asarray(contact['surface_anchor_camera'], dtype=np.float64), rotation, translation)
    normals = np.asarray(contact['surface_normal_camera'], dtype=np.float64) @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    gaps = _gaps(args.surface_gaps_mm, finger_names, args.surface_gap_mm / 1000.0)
    surface_targets_camera = np.asarray(contact['surface_anchor_camera'], dtype=np.float64) + gaps[:, None] * np.asarray(contact['surface_normal_camera'], dtype=np.float64)
    rotation_initial, translation_initial = rotation.copy(), translation.copy()
    ranges, scale = np.maximum(upper - lower, 1e-6), args.contact_scale_mm / 1000.0
    use_wrist_compensation = args.wrist_translation_limit_mm > 0 and args.wrist_rotation_limit_deg > 0
    if (args.wrist_translation_limit_mm > 0) != (args.wrist_rotation_limit_deg > 0):
        raise ValueError('Set both wrist compensation limits, or neither')
    translation_limit, rotation_limit = args.wrist_translation_limit_mm / 1000.0, np.deg2rad(args.wrist_rotation_limit_deg)

    def target_pose(delta_rotation: np.ndarray, delta_translation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = Rotation.from_rotvec(delta_rotation).as_matrix()
        current_rotation = delta @ rotation_initial
        current_translation = translation_initial + delta_translation
        return current_rotation, current_translation, _apply(surface_targets_camera, current_rotation, current_translation)

    def residual(values: np.ndarray) -> np.ndarray:
        qpos = values[:len(baseline_q)]
        if use_wrist_compensation:
            delta_translation = values[len(baseline_q):len(baseline_q) + 3]
            delta_rotation = values[len(baseline_q) + 3:]
        else:
            delta_translation, delta_rotation = np.zeros(3), np.zeros(3)
        _current_rotation, _current_translation, targets = target_pose(delta_rotation, delta_translation)
        terms = [np.sqrt(args.contact_weight) * (fk(qpos, contact_offsets)[active] - targets[active]).reshape(-1) / scale, np.sqrt(args.posture_weight) * (qpos - baseline_q) / ranges]
        if use_wrist_compensation:
            terms.extend((np.sqrt(args.wrist_position_weight) * delta_translation / translation_limit, np.sqrt(args.wrist_orientation_weight) * delta_rotation / rotation_limit))
        return np.concatenate(terms)

    if use_wrist_compensation:
        initial = np.concatenate((baseline_q, np.zeros(6)))
        lower_bounds = np.concatenate((lower + 1e-6, np.full(3, -translation_limit), np.full(3, -rotation_limit)))
        upper_bounds = np.concatenate((upper - 1e-6, np.full(3, translation_limit), np.full(3, rotation_limit)))
    else:
        initial, lower_bounds, upper_bounds = baseline_q, lower + 1e-6, upper - 1e-6
    solve = least_squares(residual, initial, bounds=(lower_bounds, upper_bounds), method='trf', max_nfev=args.max_evaluations)
    qpos = np.asarray(solve.x[:len(baseline_q)], dtype=np.float64)
    if use_wrist_compensation:
        delta_translation = np.asarray(solve.x[len(baseline_q):len(baseline_q) + 3], dtype=np.float64)
        delta_rotation = np.asarray(solve.x[len(baseline_q) + 3:], dtype=np.float64)
    else:
        delta_translation, delta_rotation = np.zeros(3), np.zeros(3)
    rotation, translation, targets = target_pose(delta_rotation, delta_translation)
    final_contacts, final_tips = fk(qpos, contact_offsets), fk(qpos, task_offsets)
    initial_targets = _apply(surface_targets_camera, rotation_initial, translation_initial)
    before, after = np.linalg.norm(baseline_contacts - initial_targets, axis=1), np.linalg.norm(final_contacts - targets, axis=1)
    by_vector = {name.lower(): value for name, value in zip(vector_names, qpos)}
    object_to_o30 = np.eye(4, dtype=np.float64)
    object_to_o30[:3, :3], object_to_o30[:3, 3] = rotation, translation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, schema_version=np.asarray(3), simulation_only=np.asarray(True), object_scale_fixed_to_one=np.asarray(True), robot=np.asarray('o30'), optimizer=np.asarray('o30_rigid_distal_pad_object_relative_contact'), retarget_backend=np.asarray(args.backend), source_contact_plan=np.asarray(str(args.contact_plan.resolve())), scale_profile=np.asarray(str(args.scale_profile.resolve()) if args.scale_profile else 'nominal_yaml_baseline'), source_canonical_grasp=np.asarray(str(state_path.resolve())), source_object_mesh=contact['source_object_mesh'], robot_joint_names=np.asarray(model_names), qpos=np.asarray([by_vector[name.lower()] for name in model_names], dtype=np.float32), vector_joint_names=np.asarray(vector_names), qpos_vector_order=qpos.astype(np.float32), qpos_initial_retarget=baseline_q.astype(np.float32), active_contact_mask=active.astype(np.uint8), active_contact_fingers=np.asarray(finger_names)[active], contact_point_alpha=alpha.astype(np.float32), contact_task_offsets_o30=contact_offsets.astype(np.float32), o30_fingertip_link_names=np.asarray(task_names), o30_fingertip_task_offsets=contact_offsets.astype(np.float32), o30_fingertip_positions_baseline=baseline_tips.astype(np.float32), o30_fingertip_positions_optimized=final_tips.astype(np.float32), o30_contact_positions_baseline=baseline_contacts.astype(np.float32), o30_contact_positions_optimized=final_contacts.astype(np.float32), contact_target_positions_o30=targets.astype(np.float32), contact_error_before_m=before.astype(np.float32), contact_error_after_m=after.astype(np.float32), camera_to_o30_rotation=rotation.astype(np.float32), camera_to_o30_translation=translation.astype(np.float32), human_to_o30_uniform_scale=np.asarray(1.0, dtype=np.float32), object_uniform_scale_in_o30_frame=np.asarray(1.0, dtype=np.float32), object_to_o30=object_to_o30.astype(np.float32), surface_gap_per_finger_m=gaps.astype(np.float32), alignment_reference=np.asarray(reference), wrist_translation_adjustment_m=delta_translation.astype(np.float32), wrist_rotation_adjustment_rad=delta_rotation.astype(np.float32), wrist_translation_limit_m=np.asarray(translation_limit, dtype=np.float32), wrist_rotation_limit_rad=np.asarray(rotation_limit, dtype=np.float32))
    report = {'simulation_only': True, 'hardware_command_generated': False, 'object_scale_fixed_to_one': True, 'method': 'rigid_no_scale_fit_then_distal_pad_surface_refinement_with_bounded_o30_wrist_compensation' if use_wrist_compensation else 'rigid_no_scale_fit_then_distal_pad_surface_refinement', 'alignment_reference': reference, 'scale_profile': str(args.scale_profile.resolve()) if args.scale_profile else 'nominal_yaml_baseline', 'active_fingers': [name for name, enabled in zip(finger_names, active) if enabled], 'optimizer_success': bool(solve.success), 'function_evaluations': int(solve.nfev), 'active_contact_error_before_mm': (before[active] * 1000).tolist(), 'active_contact_error_after_mm': (after[active] * 1000).tolist(), 'active_mean_error_before_mm': float(before[active].mean() * 1000), 'active_mean_error_after_mm': float(after[active].mean() * 1000), 'surface_gap_per_finger_mm': {name: float(gap * 1000) for name, gap in zip(finger_names, gaps)}, 'wrist_translation_adjustment_mm': (delta_translation * 1000).tolist(), 'wrist_rotation_adjustment_deg': np.rad2deg(delta_rotation).tolist(), 'wrist_translation_limit_mm': float(args.wrist_translation_limit_mm), 'wrist_rotation_limit_deg': float(args.wrist_rotation_limit_deg), 'joint_saturation_count': int(np.count_nonzero(np.minimum((qpos - lower) / ranges, (upper - qpos) / ranges) <= .05)), 'limitations': 'Rigid hand-relative geometry is simulation only; mesh collision, force closure, and hardware calibration are separate checks.'}
    report_path = args.output.with_suffix('.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Rigid no-scale O30 distal-pad plan written (simulation only)')
    print('  active contacts: ' + ', '.join(report['active_fingers']))
    print(f"  active contact mean error: {report['active_mean_error_before_mm']:.2f} -> {report['active_mean_error_after_mm']:.2f} mm")
    print(f'  output: {args.output}')


if __name__ == '__main__':
    main()
