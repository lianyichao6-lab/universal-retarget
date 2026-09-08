#!/usr/bin/env python3
"""Execute an offline AR5 + L25 trajectory against a free MuJoCo object.

This is an offline contact and lift check. The saved AR5 and L25 commands
kinematically drive the full Luban robot model while MuJoCo integrates the
object. It does not publish ROS commands or connect to hardware.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import (
    AR5_RIGHT_JOINT_NAMES,
    L25_ACTIVE_JOINT_NAMES,
    l25_qpos_to_luban_active,
)
try:  # Supports both `python tools/...py` and package-style test imports.
    from tools.luban_mujoco_viewer import (
        apply_joint_state,
        load_offline_episode,
        model_joint_qpos_addresses,
        prepare_episode_model,
        resolve_model_path,
    )
except ModuleNotFoundError:
    from luban_mujoco_viewer import (
        apply_joint_state,
        load_offline_episode,
        model_joint_qpos_addresses,
        prepare_episode_model,
        resolve_model_path,
    )




L25_MIMIC_COUPLINGS = (
    ("thumb_mcp", "thumb_dip", 1.03),
    ("index_pip", "index_dip", 0.89),
    ("middle_pip", "middle_dip", 0.89),
    ("ring_pip", "ring_dip", 0.89),
    ("pinky_pip", "pinky_dip", 0.89),
)


def apply_l25_mimic_qpos(data, addresses: dict[str, int]) -> None:
    """Mirror Luban's five right-L25 mechanical couplings into MuJoCo qpos."""
    for source_suffix, target_suffix, multiplier in L25_MIMIC_COUPLINGS:
        source = addresses.get(f"r_hand_{source_suffix}")
        target = addresses.get(f"r_hand_{target_suffix}")
        if source is None or target is None:
            raise RuntimeError(f"MuJoCo model is missing L25 mimic pair {source_suffix}->{target_suffix}")
        data.qpos[target] = data.qpos[source] * multiplier

def _body_transform(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = data.xmat[body_id].reshape(3, 3)
    result[:3, 3] = data.xpos[body_id]
    return result


def derive_luban_world_from_anydex_base(
    model_path: Path,
    arm: np.ndarray,
    flange_trajectory: Path,
    flange_body: str,
) -> tuple[np.ndarray, float, float]:
    """Estimate the fixed frame bridge from equal AR5 joint states."""
    with np.load(flange_trajectory, allow_pickle=False) as payload:
        if "T_robot_base_arm_flange" not in payload:
            raise ValueError("flange trajectory requires T_robot_base_arm_flange")
        expected = np.asarray(payload["T_robot_base_arm_flange"], dtype=np.float64)
    if expected.shape != (len(arm), 4, 4) or not np.isfinite(expected).all():
        raise ValueError("flange trajectory must be finite with shape (N, 4, 4) aligned with --arm-trajectory")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = model_joint_qpos_addresses(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flange_body)
    if body < 0:
        raise ValueError(f"Luban model does not contain flange body: {flange_body}")
    bridges = []
    for qpos, target in zip(arm, expected):
        apply_joint_state(model, data, addresses, AR5_RIGHT_JOINT_NAMES, qpos)
        mujoco.mj_forward(model, data)
        bridges.append(_body_transform(model, data, body) @ np.linalg.inv(target))
    bridges = np.asarray(bridges)
    reference = bridges[0]
    translation_variation = float(np.linalg.norm(bridges[:, :3, 3] - reference[:3, 3], axis=1).max())
    rotation_variation = float(max(
        np.arccos(np.clip((np.trace(reference[:3, :3].T @ bridge[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        for bridge in bridges
    ))
    return reference, translation_variation, rotation_variation


def localize_mesh_in_luban_world(source_mesh: Path, bridge: np.ndarray, output: Path) -> np.ndarray:
    """Bake the frame bridge into a mesh, then return its world body origin."""
    mesh = trimesh.load_mesh(source_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("object mesh must contain one triangle mesh")
    transformed = mesh.copy()
    transformed.apply_transform(bridge)
    body_position = transformed.bounds.mean(axis=0)
    transformed.apply_translation(-body_position)
    transformed.export(output)
    return np.asarray(body_position, dtype=np.float64)

def robot_qvel_addresses(model: mujoco.MjModel) -> np.ndarray:
    """Find velocity slots belonging to commanded right AR5 and L25 joints only."""
    result = []
    names = (*AR5_RIGHT_JOINT_NAMES, *(f"r_hand_{name}" for name in L25_ACTIVE_JOINT_NAMES))
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"model is missing commanded joint: {name}")
        dof_address = int(model.jnt_dofadr[joint_id])
        if dof_address < 0:
            raise RuntimeError(f"model joint has no velocity coordinate: {name}")
        result.append(dof_address)
    return np.asarray(result, dtype=np.int32)


def object_contact_count(model: mujoco.MjModel, data: mujoco.MjData, object_geom: int) -> int:
    """Count contacts between the free object and the robot/table scene."""
    return sum(
        int(data.contact[index].geom1 == object_geom or data.contact[index].geom2 == object_geom)
        for index in range(data.ncon)
    )


def set_robot_qpos(model, data, addresses, robot_qvel, arm_qpos, hand_qpos) -> None:
    """Anchor commanded joints while leaving the free object to MuJoCo."""
    apply_joint_state(model, data, addresses, AR5_RIGHT_JOINT_NAMES, arm_qpos)
    apply_joint_state(
        model,
        data,
        addresses,
        tuple(f"r_hand_{name}" for name in L25_ACTIVE_JOINT_NAMES),
        l25_qpos_to_luban_active(hand_qpos),
    )
    apply_l25_mimic_qpos(data, addresses)
    data.qvel[robot_qvel] = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Luban temporary URDF/MJCF model")
    parser.add_argument("--arm-trajectory", type=Path, required=True)
    parser.add_argument("--hand-episode", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True, help="Object mesh in AnyDex AR5-base coordinates")
    parser.add_argument("--flange-trajectory", type=Path, required=True, help="AnyDex FK NPZ for the same AR5 trajectory")
    parser.add_argument("--luban-flange-body", default="r_link7")
    parser.add_argument("--max-frame-bridge-translation-m", type=float, default=0.003)
    parser.add_argument("--max-frame-bridge-rotation-rad", type=float, default=0.003)
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ with physics object state")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--substeps", type=int, default=25, help="MuJoCo substeps per command frame")
    parser.add_argument("--settle-substeps", type=int, default=250, help="Free-object settling steps before replay")
    parser.add_argument("--min-lift-m", type=float, default=0.01)
    parser.add_argument("--max-displacement-m", type=float, default=0.25, help="Report when an object leaves the local workspace")
    parser.add_argument("--max-absolute-position-m", type=float, default=2.0, help="Reject numerically unstable object positions")
    parser.add_argument("--floor", choices=("grid", "blue", "none"), default="grid")
    parser.add_argument("--object-position", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--object-quaternion", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"))
    args = parser.parse_args()
    if (args.fps <= 0 or args.substeps <= 0 or args.settle_substeps < 0 or min(args.min_lift_m, args.max_displacement_m, args.max_frame_bridge_translation_m, args.max_frame_bridge_rotation_rad, args.max_absolute_position_m) < 0):
        parser.error("--fps and --substeps must be positive; lift/displacement thresholds must be non-negative")
    if not args.object_mesh.is_file():
        parser.error(f"--object-mesh does not exist: {args.object_mesh}")

    arm, hand, _ = load_offline_episode(args.arm_trajectory, args.hand_episode)
    model_path = resolve_model_path(args.model)
    bridge, bridge_translation_variation, bridge_rotation_variation = derive_luban_world_from_anydex_base(
        model_path, arm, args.flange_trajectory, args.luban_flange_body
    )
    if bridge_translation_variation > args.max_frame_bridge_translation_m or bridge_rotation_variation > args.max_frame_bridge_rotation_rad:
        raise RuntimeError(
            "AR5 frame bridge is not fixed enough for this trajectory: "
            f"translation variation={bridge_translation_variation:.6g} m, "
            f"rotation variation={bridge_rotation_variation:.6g} rad"
        )
    probe_model = mujoco.MjModel.from_xml_path(str(model_path))
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    floor_z = float(np.min(probe_data.geom_xpos[:, 2]) - 0.02)
    with tempfile.TemporaryDirectory(prefix="anydex_ar5_l25_physics_") as temporary:
        world_mesh = Path(temporary) / "object_in_luban_world.stl"
        object_body_position = localize_mesh_in_luban_world(args.object_mesh, bridge, world_mesh)
        object_body_position += np.asarray(args.object_position, dtype=np.float64)
        scene = prepare_episode_model(
            model_path,
            args.floor,
            floor_z,
            world_mesh,
            object_body_position,
            np.asarray(args.object_quaternion, dtype=np.float64),
            object_collision=True,
            floor_collision=True,
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        addresses = model_joint_qpos_addresses(model)
        object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "anydex_episode_object_freejoint")
        object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "anydex_episode_object_geom")
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anydex_episode_object")
        if min(object_joint, object_geom, object_body) < 0:
            raise RuntimeError("generated scene is missing its free object")
        robot_qvel = robot_qvel_addresses(model)
        if len(robot_qvel) != 23:
            raise RuntimeError(f"model exposes {len(robot_qvel)} commanded robot velocity slots; expected 23 (AR5 7 + L25 active 16)")
        model.opt.timestep = 1.0 / (args.fps * args.substeps)
        mujoco.mj_resetData(model, data)
        set_robot_qpos(model, data, addresses, robot_qvel, arm[0], hand[0])
        mujoco.mj_forward(model, data)
        for _ in range(args.settle_substeps):
            set_robot_qpos(model, data, addresses, robot_qvel, arm[0], hand[0])
            mujoco.mj_step(model, data)
        initial_position = data.xpos[object_body].copy()
        positions = np.empty((len(arm), 3), dtype=np.float32)
        contacts = np.zeros(len(arm), dtype=np.int32)
        previous_arm = arm[0].copy()
        previous_hand = hand[0].copy()
        for index in range(len(arm)):
            max_contacts = 0
            for step in range(args.substeps):
                fraction = float(step + 1) / args.substeps
                arm_qpos = previous_arm + fraction * (arm[index] - previous_arm)
                hand_qpos = previous_hand + fraction * (hand[index] - previous_hand)
                set_robot_qpos(model, data, addresses, robot_qvel, arm_qpos, hand_qpos)
                mujoco.mj_step(model, data)
                max_contacts = max(max_contacts, object_contact_count(model, data, object_geom))
            positions[index] = data.xpos[object_body]
            contacts[index] = max_contacts
            previous_arm = arm[index].copy()
            previous_hand = hand[index].copy()

    finite_positions = bool(np.isfinite(positions).all())
    displacement = np.linalg.norm(positions.astype(np.float64) - initial_position[None], axis=1)
    max_displacement = float(np.max(displacement)) if finite_positions else float("inf")
    physics_stable = bool(
        finite_positions and np.max(np.abs(positions.astype(np.float64))) <= args.max_absolute_position_m
    )
    object_left_workspace = bool(max_displacement > args.max_displacement_m)
    height_delta = float(positions[-1, 2] - initial_position[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=np.arange(len(arm), dtype=np.float64) / args.fps,
        ar5_qpos=arm.astype(np.float32),
        l25_qpos=hand.astype(np.float32),
        object_position=positions,
        object_contact_count=contacts,
    )
    report = {
        "frames": int(len(arm)),
        "T_luban_world_anydex_ar5_base": bridge.tolist(),
        "frame_bridge_translation_variation_m": bridge_translation_variation,
        "frame_bridge_rotation_variation_rad": bridge_rotation_variation,
        "object_initial_body_position_luban_world": object_body_position.tolist(),
        "fps": args.fps,
        "substeps": args.substeps,
        "settle_substeps": args.settle_substeps,
        "object_contact_observed": bool(np.any(contacts > 0)),
        "max_object_contact_count": int(contacts.max()),
        "object_height_delta_m": height_delta,
        "min_lift_m": args.min_lift_m,
        "max_displacement_m": args.max_displacement_m,
        "max_observed_displacement_m": max_displacement,
        "max_absolute_position_m": args.max_absolute_position_m,
        "physics_stable": physics_stable,
        "object_left_workspace": object_left_workspace,
        "lift_success": bool(physics_stable and np.any(contacts > 0) and height_delta >= args.min_lift_m),
        "failure_reason": (
            "numerically_unstable_object_motion" if not physics_stable
            else "object_left_workspace" if object_left_workspace
            else "no_sustained_lift"
        ),
        "interpretation": "offline MuJoCo contact/lift check; robot joints are command-anchored",
        "hardware_command_generated": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
