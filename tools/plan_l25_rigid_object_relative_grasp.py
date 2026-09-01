#!/usr/bin/env python3
"""Create a no-scale L25 object-relative grasp plan from HUG geometry.

The reconstructed object remains metric: only a rigid camera-to-L25 rotation
and translation are estimated.  They are fitted from HUG fingertip positions
to the baseline L25 FK fingertips, then L25 joints are bounded-refined toward
the corresponding fixed object-relative fingertip targets.

This is a hand-relative simulation embedding, not camera-to-robot calibration
and not a real-hardware command generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.l25_retarget_backend import BACKENDS, VECTOR_CONFIG, retarget_l25_static


ROOT = Path(__file__).resolve().parents[1]
L25_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"


def _rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t with target ~= source @ R.T + t, explicitly without scale."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3 or source.shape[1] != 3:
        raise ValueError("Rigid fit requires matching N x 3 arrays with N >= 3")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _apply(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation[None]


def _load_contact_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]).copy() for key in data.files}
    for key in ("source_canonical_grasp", "near_surface", "fingertip_positions_camera", "surface_anchor_camera"):
        if key not in result:
            raise ValueError(f"Contact plan missing {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="vector")
    parser.add_argument("--vector-config", type=Path, default=VECTOR_CONFIG,
                        help="Audited L25 FK/task-point config; also the default Vector backend config.")
    parser.add_argument("--config", type=Path,
                        help="Optional native Vector/Adaptive YAML override; unsupported for dex backends.")
    parser.add_argument("--dex-scaling", type=float)
    parser.add_argument("--dex-project-dist", type=float)
    parser.add_argument("--dex-escape-dist", type=float)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.12)
    parser.add_argument("--contact-scale-mm", type=float, default=10.0)
    parser.add_argument("--surface-gap-mm", type=float, default=0.0,
                        help="Move active fingertip targets outward from the mesh surface.")
    parser.add_argument(
        "--surface-gaps-mm",
        type=str,
        help="Optional per-finger overrides, for example thumb=11,index=4,middle=4.",
    )
    parser.add_argument("--contact-fingers", type=str,
                        help="Comma-separated subset of near-surface fingers, e.g. thumb,index,middle.")
    parser.add_argument("--max-evaluations", type=int, default=160)
    args = parser.parse_args()
    if (args.contact_weight <= 0 or args.posture_weight < 0 or args.contact_scale_mm <= 0
            or args.surface_gap_mm < 0 or args.max_evaluations <= 0):
        raise ValueError("Invalid optimization settings")
    if args.config is not None and args.backend in {"dexpilot", "joint_angle"}:
        raise ValueError("--config is only supported for native Vector/Adaptive backends")
    if any(value is not None and value <= 0 for value in
           (args.dex_scaling, args.dex_project_dist, args.dex_escape_dist)):
        raise ValueError("Dex backend overrides must be positive")
    if (args.dex_project_dist is not None and args.dex_escape_dist is not None
            and args.dex_escape_dist < args.dex_project_dist):
        raise ValueError("--dex-escape-dist must be >= --dex-project-dist")

    contact = _load_contact_plan(args.contact_plan)
    active = np.asarray(contact["near_surface"], dtype=np.uint8).astype(bool)
    finger_names = [str(name) for name in contact.get("finger_names", [])]
    if len(finger_names) != len(active):
        raise ValueError("Contact plan finger names do not match fingertip mask")
    if args.contact_fingers:
        requested = {name.strip().lower() for name in args.contact_fingers.split(",") if name.strip()}
        known = {name.lower() for name in finger_names}
        unknown = requested - known
        if unknown:
            raise ValueError("Unknown contact finger(s): " + ", ".join(sorted(unknown)))
        active &= np.asarray([name.lower() in requested for name in finger_names], dtype=bool)
    if active.sum() < 3:
        raise ValueError("Need at least three near-surface HUG fingertips for a rigid object placement")
    state_path = Path(str(contact["source_canonical_grasp"].item()))
    state = load_canonical_grasp_state(state_path)
    human_tips = np.asarray(contact["fingertip_positions_camera"], dtype=np.float64)
    if human_tips.shape != (5, 3):
        raise ValueError("Contact plan fingertip positions must be 5 x 3")

    baseline = retarget_l25_static(
        args.backend, state.keypoints_for_retargeting(), native_config=args.config,
        geometry_config=args.vector_config, dex_scaling=args.dex_scaling,
        dex_project_dist=args.dex_project_dist, dex_escape_dist=args.dex_escape_dist,
    )
    retargeter = baseline.geometry_retargeter
    robot = retargeter.optimizer.robot
    vector_names = baseline.joint_names
    task_names = list(retargeter.optimizer.task_link_names)
    task_ids = [robot.get_link_index(name) for name in task_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    baseline_q = np.asarray(baseline.qpos, dtype=np.float64)
    if task_offsets.shape != (5, 3) or baseline_q.shape != (len(vector_names),):
        raise ValueError("Unexpected audited L25 task geometry or backend qpos")

    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    by_model_name = {str(name).lower(): i for i, name in enumerate(model_names) if name is not None}
    vector_model_indices = np.asarray([by_model_name[name.lower()] for name in vector_names], dtype=np.int64)
    lower, upper = model.jnt_range[vector_model_indices, 0], model.jnt_range[vector_model_indices, 1]
    baseline_q = np.clip(baseline_q, lower + 1e-6, upper - 1e-6)

    def fk(q: np.ndarray) -> np.ndarray:
        return robot.compute_points_batch(np.asarray(q, dtype=np.float64), task_ids, task_offsets)

    baseline_tips = fk(baseline_q)
    rotation, translation = _rigid(human_tips[active], baseline_tips[active])
    # These are the HUG fingers' original locations relative to the unchanged object,
    # expressed in the L25 hand-relative simulation frame.
    desired_tips = _apply(human_tips, rotation, translation)
    anchors_l25 = _apply(np.asarray(contact["surface_anchor_camera"], dtype=np.float64), rotation, translation)
    ranges = np.maximum(upper - lower, 1e-6)
    contact_scale = args.contact_scale_mm / 1000.0
    surface_gap = args.surface_gap_mm / 1000.0
    normals_l25 = np.asarray(contact["surface_normal_camera"], dtype=np.float64) @ rotation.T
    normals_l25 /= np.maximum(np.linalg.norm(normals_l25, axis=1, keepdims=True), 1e-12)
    surface_gaps = np.full(len(finger_names), surface_gap, dtype=np.float64)
    if args.surface_gaps_mm:
        known = {name.lower(): index for index, name in enumerate(finger_names)}
        for item in args.surface_gaps_mm.split(","):
            name, separator, value = item.strip().partition("=")
            if not separator or name.lower() not in known:
                raise ValueError("--surface-gaps-mm entries must be finger=millimeters")
            gap_mm = float(value)
            if gap_mm < 0:
                raise ValueError("--surface-gaps-mm values must be non-negative")
            surface_gaps[known[name.lower()]] = gap_mm / 1000.0
    contact_targets = desired_tips.copy()
    contact_targets[active] = anchors_l25[active] + surface_gaps[active, None] * normals_l25[active]

    def residual(q: np.ndarray) -> np.ndarray:
        return np.concatenate((
            np.sqrt(args.contact_weight) * (fk(q)[active] - contact_targets[active]).reshape(-1) / contact_scale,
            np.sqrt(args.posture_weight) * (q - baseline_q) / ranges,
        ))

    solve = least_squares(residual, baseline_q, bounds=(lower + 1e-6, upper - 1e-6),
                          method="trf", max_nfev=args.max_evaluations)
    q_vector = np.asarray(solve.x, dtype=np.float64)
    final_tips = fk(q_vector)
    before = np.linalg.norm(baseline_tips - contact_targets, axis=1)
    after = np.linalg.norm(final_tips - contact_targets, axis=1)
    q_model = np.asarray([q_vector[vector_names.index(str(name))] for name in model_names], dtype=np.float32)
    margins = np.minimum((q_vector - lower) / ranges, (upper - q_vector) / ranges)
    object_to_l25 = np.eye(4, dtype=np.float64)
    object_to_l25[:3, :3] = rotation
    object_to_l25[:3, 3] = translation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(2, dtype=np.int64),
        simulation_only=np.asarray(True),
        object_scale_fixed_to_one=np.asarray(True),
        source_contact_plan=np.asarray(str(args.contact_plan.resolve())),
        source_canonical_grasp=np.asarray(str(state_path.resolve())),
        robot=np.asarray("l25"),
        optimizer=np.asarray("l25_rigid_object_relative_contact"),
        retarget_backend=np.asarray(args.backend),
        robot_joint_names=np.asarray(model_names), qpos=q_model,
        vector_joint_names=np.asarray(vector_names), qpos_vector_order=q_vector.astype(np.float32),
        active_contact_mask=active.astype(np.uint8),
        l25_fingertip_positions_baseline=baseline_tips.astype(np.float32),
        l25_fingertip_positions_optimized=final_tips.astype(np.float32),
        desired_fingertip_positions_l25=desired_tips.astype(np.float32),
        contact_target_positions_l25=contact_targets.astype(np.float32),
        surface_normal_positions_l25=normals_l25.astype(np.float32),
        surface_gap_m=np.asarray(surface_gap, dtype=np.float32),
        surface_gap_per_finger_m=surface_gaps.astype(np.float32),
        surface_anchor_positions_l25=anchors_l25.astype(np.float32),
        fingertip_error_before_m=before.astype(np.float32), fingertip_error_after_m=after.astype(np.float32),
        camera_to_l25_rotation=rotation.astype(np.float32), camera_to_l25_translation=translation.astype(np.float32),
        human_to_l25_uniform_scale=np.asarray(1.0, dtype=np.float32),
        object_uniform_scale_in_l25_frame=np.asarray(1.0, dtype=np.float32),
        object_to_l25=object_to_l25.astype(np.float32),
    )
    report = {
        "simulation_only": True,
        "object_scale_fixed_to_one": True,
        "method": "rigid_no_scale_fit_then_surface_normal_clearance_refinement",
        "retarget_backend": args.backend,
        "source_contact_plan": str(args.contact_plan.resolve()),
        "active_fingertip_count": int(active.sum()),
        "active_fingers": [name for name, enabled in zip(finger_names, active) if enabled],
        "optimizer_success": bool(solve.success), "function_evaluations": int(solve.nfev),
        "active_fingertip_error_before_mm": (before[active] * 1000.0).tolist(),
        "active_fingertip_error_after_mm": (after[active] * 1000.0).tolist(),
        "active_mean_error_before_mm": float(before[active].mean() * 1000.0),
        "active_mean_error_after_mm": float(after[active].mean() * 1000.0),
        "surface_gap_mm": float(args.surface_gap_mm),
        "surface_gap_per_finger_mm": {
            name: float(gap * 1000.0) for name, gap in zip(finger_names, surface_gaps)
        },
        "joint_limit_violations": int(np.count_nonzero((q_vector < lower) | (q_vector > upper))),
        "joint_saturation_count": int(np.count_nonzero(margins <= 0.05)),
        "limitations": "This preserves object metric mesh scale and HUG hand-object geometry only in a hand-relative simulation frame. Full-link collision and force closure are evaluated separately; real camera-to-robot calibration is not supplied.",
        "hardware_command_generated": False,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Rigid no-scale L25 object-relative plan written (simulation only)")
    print(f"  initial retarget backend: {args.backend}")
    print(f"  active fingertip mean error: {report['active_mean_error_before_mm']:.2f} -> {report['active_mean_error_after_mm']:.2f} mm")
    print(f"  saturation: {report['joint_saturation_count']}; violations: {report['joint_limit_violations']}")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
