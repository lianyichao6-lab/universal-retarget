#!/usr/bin/env python3
"""Plan an offline AR5 + L25 grasp against the exact Luban MuJoCo arm model.

This is for simulation validation only.  It keeps the HUG/L25 grasp request
in the AnyDex AR5-base frame, calibrates that frame once at the start pose,
then solves the right-arm trajectory using the same `r_link7` geometry that
the physical MuJoCo episode runner will execute.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from anydexretarget.arm_fk import AR5ForwardKinematics
from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES
try:  # Support both `python tools/...py` and package-style test imports.
    from tools.luban_mujoco_viewer import apply_joint_state, model_joint_qpos_addresses, resolve_model_path
    from tools.plan_ar5_l25_cumotion import _hand_timeline, _read_request
except ModuleNotFoundError:
    from luban_mujoco_viewer import apply_joint_state, model_joint_qpos_addresses, resolve_model_path
    from plan_ar5_l25_cumotion import _hand_timeline, _read_request


L25_FLEXION_QPOS_NAMES = (
    "thumb_mcp",
    "index_mcp_pitch", "index_pip",
    "middle_mcp_pitch", "middle_pip",
    "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_pitch", "pinky_pip",
)


def apply_l25_flexion_margin(final_qpos: np.ndarray, model: mujoco.MjModel, margin_rad: float) -> np.ndarray:
    """Close active L25 flexion joints by a bounded amount for physical validation."""
    result = np.asarray(final_qpos, dtype=np.float64).copy()
    if margin_rad == 0:
        return result
    for source_name in L25_FLEXION_QPOS_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r_hand_{source_name}")
        if joint_id < 0:
            raise RuntimeError(f"Luban model is missing L25 joint r_hand_{source_name}")
        index = L25_QPOS_JOINTS.index(source_name)
        result[index] = min(result[index] + margin_rad, model.jnt_range[joint_id, 1])
    return result


def _body_transform(data: mujoco.MjData, body_id: int) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def _set_arm(model, data, addresses: dict[str, int], qpos: np.ndarray) -> None:
    apply_joint_state(model, data, addresses, AR5_RIGHT_JOINT_NAMES, qpos)
    mujoco.mj_forward(model, data)


def _solve_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: dict[str, int],
    body_id: int,
    target: np.ndarray,
    seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, float, float, int]:
    def residual(qpos: np.ndarray) -> np.ndarray:
        _set_arm(model, data, addresses, qpos)
        current = _body_transform(data, body_id)
        position = (current[:3, 3] - target[:3, 3]) * 10.0
        orientation = Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).as_rotvec()
        return np.concatenate((position, orientation))

    solution = least_squares(
        residual,
        np.clip(seed, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    _set_arm(model, data, addresses, solution.x)
    current = _body_transform(data, body_id)
    position_error = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    orientation_error = float(np.linalg.norm(Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).as_rotvec()))
    if position_error > 0.002 or orientation_error > 0.0524:
        raise RuntimeError(
            "Luban-model AR5 IK failed: "
            f"position_error_m={position_error:.6f}, orientation_error_rad={orientation_error:.6f}"
        )
    return np.asarray(solution.x, dtype=np.float64), position_error, orientation_error, int(solution.nfev)


def _joint_segment(start: np.ndarray, goal: np.ndarray, max_step_rad: float) -> np.ndarray:
    steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_step_rad)))
    return np.linspace(start, goal, steps + 1, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Current Luban MuJoCo robot URDF/MJCF, or 'latest'.")
    parser.add_argument("--source-urdf", type=Path, required=True, help="The AnyDex AR5 URDF used by the grasp request.")
    parser.add_argument("--start-joints", type=float, nargs=7, required=True)
    parser.add_argument("--arm-output", type=Path, required=True)
    parser.add_argument("--l25-output", type=Path, required=True)
    parser.add_argument("--flange-output", type=Path, required=True)
    parser.add_argument("--flange-body", default="r_link7")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--close-frames", type=int, default=12)
    parser.add_argument("--close-hold-frames", type=int, default=10)
    parser.add_argument("--lift-hold-frames", type=int, default=10)
    parser.add_argument("--close-flexion-margin-rad", type=float, default=0.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--max-nfev", type=int, default=1000)
    args = parser.parse_args()
    if args.fps <= 0 or args.close_frames < 1 or args.close_hold_frames < 0 or args.lift_hold_frames < 0 or args.close_flexion_margin_rad < 0 or args.max_joint_step_rad <= 0 or args.max_nfev < 1:
        parser.error("fps, close frames, joint step and max_nfev must be positive; hold frames must be non-negative")

    request = _read_request(args.request)
    start = np.asarray(args.start_joints, dtype=np.float64)
    model_path = resolve_model_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = model_joint_qpos_addresses(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.flange_body)
    if body_id < 0:
        raise ValueError(f"Luban model has no flange body {args.flange_body!r}")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in AR5_RIGHT_JOINT_NAMES]
    if min(joint_ids) < 0:
        raise ValueError("Luban model is missing one or more right AR5 joints")
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids], dtype=np.float64)
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids], dtype=np.float64)

    source_fk = AR5ForwardKinematics(args.source_urdf)
    _set_arm(model, data, addresses, start)
    bridge = _body_transform(data, body_id) @ np.linalg.inv(source_fk.flange_transform(start))

    arm_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    diagnostics: dict[str, dict[str, float | int]] = {}
    current = start
    for phase, key in (("pregrasp", "T_robot_base_arm_flange_pregrasp"), ("approach", "T_robot_base_arm_flange_target")):
        target = bridge @ np.asarray(request[key], dtype=np.float64)
        solution, position_error, orientation_error, iterations = _solve_pose(
            model, data, addresses, body_id, target, current, lower, upper, args.max_nfev
        )
        segment = _joint_segment(current, solution, args.max_joint_step_rad)
        if arm_parts:
            segment = segment[1:]
        arm_parts.append(segment)
        phase_parts.append(np.full(len(segment), phase))
        diagnostics[phase] = {"position_error_m": position_error, "orientation_error_rad": orientation_error, "iterations": iterations}
        current = solution
    arm_parts.append(np.repeat(current[None], args.close_frames, axis=0))
    phase_parts.append(np.full(args.close_frames, "close"))
    if args.close_hold_frames:
        arm_parts.append(np.repeat(current[None], args.close_hold_frames, axis=0))
        phase_parts.append(np.full(args.close_hold_frames, "close_hold"))
    target = bridge @ np.asarray(request["T_robot_base_arm_flange_lift"], dtype=np.float64)
    solution, position_error, orientation_error, iterations = _solve_pose(
        model, data, addresses, body_id, target, current, lower, upper, args.max_nfev
    )
    segment = _joint_segment(current, solution, args.max_joint_step_rad)[1:]
    arm_parts.append(segment)
    phase_parts.append(np.full(len(segment), "lift"))
    diagnostics["lift"] = {"position_error_m": position_error, "orientation_error_rad": orientation_error, "iterations": iterations}
    current = solution
    if args.lift_hold_frames:
        arm_parts.append(np.repeat(current[None], args.lift_hold_frames, axis=0))
        phase_parts.append(np.full(args.lift_hold_frames, "lift_hold"))
    arm = np.concatenate(arm_parts, axis=0)
    phase = np.concatenate(phase_parts)
    final_hand_qpos = apply_l25_flexion_margin(
        np.asarray(request["l25_qpos"], dtype=np.float64), model, args.close_flexion_margin_rad
    ).astype(np.float32)
    hand, _ = _hand_timeline(
        phase,
        final_hand_qpos,
        np.asarray(request["l25_preshape_positions"], dtype=np.float32),
        args.close_frames,
    )

    source_frame_transforms = []
    for qpos in arm:
        _set_arm(model, data, addresses, qpos)
        source_frame_transforms.append(np.linalg.inv(bridge) @ _body_transform(data, body_id))
    source_frame_transforms = np.asarray(source_frame_transforms)
    timestamps = np.arange(len(arm), dtype=np.float64) / args.fps
    args.arm_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arm_output, joint_names=np.asarray(AR5_RIGHT_JOINT_NAMES), positions=arm.astype(np.float32), timestamps=timestamps, phase=phase, planner=np.asarray("luban_mujoco_numerical_ik"))
    with args.l25_output.open("wb") as stream:
        pickle.dump([{"target": target, "phase": str(name)} for target, name in zip(hand, phase)], stream)
    np.savez_compressed(args.flange_output, T_robot_base_arm_flange=source_frame_transforms, arm_positions=arm.astype(np.float32), joint_names=np.asarray(AR5_RIGHT_JOINT_NAMES), timestamps=timestamps, frame_id=np.asarray("r_base_link"), flange_frame=np.asarray(args.flange_body))
    report = {
        "frames": int(len(arm)), "fps": args.fps, "planner": "luban_mujoco_numerical_ik", "model": str(model_path),
        "flange_body": args.flange_body, "close_flexion_margin_rad": args.close_flexion_margin_rad, "close_hold_frames": args.close_hold_frames, "lift_hold_frames": args.lift_hold_frames, "T_luban_world_anydex_ar5_base": bridge.tolist(), "ik": diagnostics,
        "planning_only": True, "hardware_command_generated": False,
    }
    args.arm_output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
