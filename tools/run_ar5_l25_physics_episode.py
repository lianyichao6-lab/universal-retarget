#!/usr/bin/env python3
"""Execute an offline AR5 + L25 trajectory against a free MuJoCo object.

This is an offline contact and lift check. The saved AR5 and L25 commands
kinematically drive the full Luban robot model while MuJoCo integrates the
object. It does not publish ROS commands or connect to hardware.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
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


MUJOCO_WARNING_NAMES = {
    int(mujoco.mjtWarning.mjWARN_INERTIA): "inertia",
    int(mujoco.mjtWarning.mjWARN_CONTACTFULL): "contact_full",
    int(mujoco.mjtWarning.mjWARN_CNSTRFULL): "constraint_full",
    int(mujoco.mjtWarning.mjWARN_BADQPOS): "bad_qpos",
    int(mujoco.mjtWarning.mjWARN_BADQVEL): "bad_qvel",
    int(mujoco.mjtWarning.mjWARN_BADQACC): "bad_qacc",
    int(mujoco.mjtWarning.mjWARN_BADCTRL): "bad_ctrl",
}
CRITICAL_MUJOCO_WARNINGS = frozenset(MUJOCO_WARNING_NAMES)


def _is_robot_hinge(name: str | None, joint_type: int) -> bool:
    return bool(
        name
        and name.startswith(("r_joint_", "l_joint_", "r_hand_", "l_hand_"))
        and joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)
    )


def robot_hinge_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return all robot hinge joints, excluding the free object joint."""
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if _is_robot_hinge(name, int(model.jnt_type[joint_id])):
            names.append(name)
    if len(names) != 56:
        raise RuntimeError(f"expected 56 robot hinges (dual AR5 + dual L25), got {len(names)}")
    return tuple(names)


def _servo_parameters(joint_name: str, arm_servo: tuple[float, float, float], hand_servo: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return kp, kv and symmetric force limit for a robot position servo."""
    return hand_servo if "_hand_" in joint_name else arm_servo


def add_position_servos(scene_path: Path, source_model: mujoco.MjModel, joint_names: tuple[str, ...], arm_servo: tuple[float, float, float] = (1000.0, 100.0, 600.0), hand_servo: tuple[float, float, float] = (12.0, 0.25, 3.0)) -> None:
    """Inject bounded position servos into the temporary MuJoCo scene only."""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"scene is missing robot joint {joint_name}")
        lower, upper = source_model.jnt_range[joint_id]
        kp, kv, force_limit = _servo_parameters(joint_name, arm_servo, hand_servo)
        ET.SubElement(actuators, "position", {
            "name": f"anydex_servo__{joint_name}",
            "joint": joint_name,
            "kp": f"{kp:.9g}",
            "kv": f"{kv:.9g}",
            "ctrllimited": "true",
            "ctrlrange": f"{lower:.9g} {upper:.9g}",
            "forcelimited": "true",
            "forcerange": f"{-force_limit:.9g} {force_limit:.9g}",
        })
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)


def configure_robot_dynamics(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> None:
    """Add conservative damping and rotor inertia to the generated robot scene."""
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_address = int(model.jnt_dofadr[joint_id])
        if "_hand_" in joint_name:
            model.dof_damping[dof_address] = 0.01
            model.dof_armature[dof_address] = 0.0002
        else:
            model.dof_damping[dof_address] = 1.0
            model.dof_armature[dof_address] = 0.02


def configure_object_interaction_collisions(model: mujoco.MjModel, object_geom: int) -> None:
    """Keep object contacts while disabling imported robot self and table contacts."""
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(("r_", "l_")):
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 1
        elif body_id == 0:
            model.geom_contype[geom_id] = 4
            model.geom_conaffinity[geom_id] = 1
    model.geom_contype[object_geom] = 1
    model.geom_conaffinity[object_geom] = 2


def servo_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for joint_name in joint_names:
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"anydex_servo__{joint_name}"
        )
        if actuator_id < 0:
            raise RuntimeError(f"generated scene is missing servo for {joint_name}")
        result[joint_name] = actuator_id
    return result


def apply_l25_mimic_values(values: np.ndarray, addresses: dict[str, int]) -> None:
    """Apply the five L25 mechanical couplings to a qpos-like vector."""
    for source_suffix, target_suffix, multiplier in L25_MIMIC_COUPLINGS:
        source = addresses.get(f"r_hand_{source_suffix}")
        target = addresses.get(f"r_hand_{target_suffix}")
        if source is None or target is None:
            raise RuntimeError(f"MuJoCo model is missing L25 mimic pair {source_suffix}->{target_suffix}")
        values[target] = values[source] * multiplier


def apply_l25_mimic_qpos(data, addresses: dict[str, int]) -> None:
    """Mirror Luban mechanical couplings into MuJoCo qpos."""
    apply_l25_mimic_values(data.qpos, addresses)

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

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def load_phases(arm_path: Path, hand_path: Path, frame_count: int) -> np.ndarray:
    """Load one synchronized phase label per command frame."""
    with np.load(arm_path, allow_pickle=False) as arm_payload, np.load(hand_path, allow_pickle=False) as hand_payload:
        if "phase" not in arm_payload or "phase" not in hand_payload:
            raise ValueError("AR5 and L25 trajectories must both include phase labels")
        arm_phase = np.asarray(arm_payload["phase"], dtype=str)
        hand_phase = np.asarray(hand_payload["phase"], dtype=str)
    if arm_phase.shape != (frame_count,) or hand_phase.shape != (frame_count,):
        raise ValueError("phase labels must align with trajectory frames")
    if not np.array_equal(arm_phase, hand_phase):
        raise ValueError("AR5 and L25 phase labels must match")
    return arm_phase


def right_command_target(initial_qpos: np.ndarray, addresses: dict[str, int], arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> np.ndarray:
    """Build a whole-robot servo target while holding the unused left side fixed."""
    target = initial_qpos.copy()
    for joint_name, value in zip(AR5_RIGHT_JOINT_NAMES, arm_qpos):
        target[addresses[joint_name]] = value
    for joint_name, value in zip(L25_ACTIVE_JOINT_NAMES, l25_qpos_to_luban_active(hand_qpos)):
        target[addresses[f"r_hand_{joint_name}"]] = value
    apply_l25_mimic_values(target, addresses)
    return target


def set_servo_targets(data: mujoco.MjData, addresses: dict[str, int], servos: dict[str, int], target_qpos: np.ndarray) -> None:
    for joint_name, actuator_id in servos.items():
        data.ctrl[actuator_id] = target_qpos[addresses[joint_name]]


def right_hand_qpos_from_vector(qpos: np.ndarray, addresses: dict[str, int]) -> np.ndarray:
    return np.asarray([
        qpos[addresses[f"r_hand_{'thumb_dip' if name == 'thumb_ip' else name}"]]
        for name in L25_QPOS_JOINTS
    ], dtype=np.float64)


def read_right_state(data: mujoco.MjData, addresses: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    arm = np.asarray([data.qpos[addresses[name]] for name in AR5_RIGHT_JOINT_NAMES], dtype=np.float64)
    return arm, right_hand_qpos_from_vector(data.qpos, addresses)


def warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {name: int(data.warning[index].number) for index, name in MUJOCO_WARNING_NAMES.items()}


def has_critical_warning(data: mujoco.MjData) -> bool:
    return any(data.warning[index].number > 0 for index in CRITICAL_MUJOCO_WARNINGS)


def object_contact_diagnostics(model: mujoco.MjModel, data: mujoco.MjData, object_geom: int) -> tuple[int, frozenset[str], bool, int]:
    """Return hand contacts, distinct fingers, support contact and arm contact."""
    hand_contacts = 0
    fingers: set[str] = set()
    support_contact = False
    arm_contacts = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 != object_geom and contact.geom2 != object_geom:
            continue
        other_geom = contact.geom2 if contact.geom1 == object_geom else contact.geom1
        body_id = int(model.geom_bodyid[other_geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("r_hand_"):
            hand_contacts += 1
            for finger in FINGER_NAMES:
                if body_name.startswith(f"r_hand_{finger}_"):
                    fingers.add(finger)
                    break
        elif body_id == 0:
            support_contact = True
        elif body_name.startswith("r_"):
            arm_contacts += 1
    return hand_contacts, frozenset(fingers), support_contact, arm_contacts


def object_relative_to_body(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, object_position: np.ndarray) -> np.ndarray:
    transform = _body_transform(model, data, body_id)
    return transform[:3, :3].T @ (object_position - transform[:3, 3])


def failure_reason(physics_stable: bool, tracks_command: bool, hand_contact_observed: bool, enough_fingers: bool, arm_contact_observed: bool, lift_contact_ratio: float, final_support_contact: bool, lift_delta: float, relative_drift: float, object_left_workspace: bool, min_lift_m: float, max_relative_drift_m: float) -> str:
    if not physics_stable:
        return "mujoco_numerical_instability"
    if not tracks_command:
        return "robot_tracking_error"
    if object_left_workspace:
        return "object_left_workspace"
    if not hand_contact_observed:
        return "no_right_hand_contact"
    if not enough_fingers:
        return "insufficient_finger_contacts"
    if arm_contact_observed:
        return "unexpected_arm_object_contact"
    if lift_contact_ratio < 0.5:
        return "contact_lost_during_lift"
    if final_support_contact:
        return "object_still_supported"
    if lift_delta < min_lift_m:
        return "no_sustained_lift"
    if relative_drift > max_relative_drift_m:
        return "object_slipped_from_hand"
    return "success"


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
    parser.add_argument("--command-time-scale", type=float, default=1.0, help="Physical seconds per trajectory second")
    parser.add_argument("--substeps", type=int, default=25, help="MuJoCo substeps per command frame")
    parser.add_argument("--settle-substeps", type=int, default=250, help="Free-object settling steps before replay")
    parser.add_argument("--min-lift-m", type=float, default=0.01)
    parser.add_argument("--max-relative-drift-m", type=float, default=0.03)
    parser.add_argument("--max-arm-tracking-error-rad", type=float, default=0.05)
    parser.add_argument("--max-hand-tracking-error-rad", type=float, default=0.12)
    parser.add_argument("--arm-servo-kp", type=float, default=1000.0)
    parser.add_argument("--arm-servo-kv", type=float, default=100.0)
    parser.add_argument("--arm-servo-force-nm", type=float, default=600.0)
    parser.add_argument("--hand-servo-kp", type=float, default=12.0)
    parser.add_argument("--hand-servo-kv", type=float, default=0.25)
    parser.add_argument("--hand-servo-force-nm", type=float, default=3.0)
    parser.add_argument("--max-displacement-m", type=float, default=0.25, help="Report when an object leaves the local workspace")
    parser.add_argument("--max-absolute-position-m", type=float, default=2.0, help="Reject numerically unstable object positions")
    parser.add_argument("--floor", choices=("grid", "blue", "none"), default="grid")
    parser.add_argument("--object-position", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--object-quaternion", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"))
    args = parser.parse_args()
    thresholds = (args.command_time_scale, args.arm_servo_kp, args.arm_servo_kv, args.arm_servo_force_nm, args.hand_servo_kp, args.hand_servo_kv, args.hand_servo_force_nm, args.min_lift_m, args.max_relative_drift_m, args.max_arm_tracking_error_rad, args.max_hand_tracking_error_rad, args.max_displacement_m, args.max_frame_bridge_translation_m, args.max_frame_bridge_rotation_rad, args.max_absolute_position_m)
    if args.fps <= 0 or args.substeps <= 0 or args.settle_substeps < 0 or min(thresholds) < 0:
        parser.error("fps/substeps must be positive and all thresholds must be non-negative")
    if not args.object_mesh.is_file():
        parser.error(f"--object-mesh does not exist: {args.object_mesh}")

    arm_command, hand_command, _ = load_offline_episode(args.arm_trajectory, args.hand_episode)
    phases = load_phases(args.arm_trajectory, args.hand_episode, len(arm_command))
    model_path = resolve_model_path(args.model)
    bridge, bridge_translation_variation, bridge_rotation_variation = derive_luban_world_from_anydex_base(
        model_path, arm_command, args.flange_trajectory, args.luban_flange_body
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
        world_mesh_output = args.output.with_name(f"{args.output.stem}_object_world.stl")
        world_mesh_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(world_mesh, world_mesh_output)
        object_body_position += np.asarray(args.object_position, dtype=np.float64)
        scene = prepare_episode_model(
            model_path, args.floor, floor_z, world_mesh, object_body_position,
            np.asarray(args.object_quaternion, dtype=np.float64), object_collision=True, floor_collision=True,
        )
        source_model = mujoco.MjModel.from_xml_path(str(scene))
        controlled_joints = robot_hinge_names(source_model)
        add_position_servos(
            scene, source_model, controlled_joints,
            (args.arm_servo_kp, args.arm_servo_kv, args.arm_servo_force_nm),
            (args.hand_servo_kp, args.hand_servo_kv, args.hand_servo_force_nm),
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = args.command_time_scale / (args.fps * args.substeps)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        configure_robot_dynamics(model, controlled_joints)
        object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "anydex_episode_object_geom")
        configure_object_interaction_collisions(model, object_geom)
        data = mujoco.MjData(model)
        addresses = model_joint_qpos_addresses(model)
        servos = servo_addresses(model, controlled_joints)
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anydex_episode_object")
        flange_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.luban_flange_body)
        if min(object_geom, object_body, flange_body) < 0:
            raise RuntimeError("generated scene is missing its object or right AR5 flange body")

        mujoco.mj_resetData(model, data)
        initial_target = right_command_target(data.qpos, addresses, arm_command[0], hand_command[0])
        data.qpos[:] = initial_target
        data.qvel[:] = 0.0
        set_servo_targets(data, addresses, servos, initial_target)
        mujoco.mj_forward(model, data)
        unstable = False
        for _ in range(args.settle_substeps):
            set_servo_targets(data, addresses, servos, initial_target)
            mujoco.mj_step(model, data)
            if has_critical_warning(data):
                unstable = True
                break
        initial_position = data.xpos[object_body].copy()
        _, _, initial_support_contact, _ = object_contact_diagnostics(model, data, object_geom)

        frame_count = len(arm_command)
        positions = np.empty((frame_count, 3), dtype=np.float64)
        quaternions = np.empty((frame_count, 4), dtype=np.float64)
        relative_positions = np.empty((frame_count, 3), dtype=np.float64)
        arm_actual = np.empty_like(arm_command, dtype=np.float64)
        hand_actual = np.empty_like(hand_command, dtype=np.float64)
        hand_contacts = np.zeros(frame_count, dtype=np.int32)
        finger_counts = np.zeros(frame_count, dtype=np.int32)
        support_contacts = np.zeros(frame_count, dtype=bool)
        arm_contacts = np.zeros(frame_count, dtype=np.int32)
        executed_frames = 0
        previous_target = initial_target
        if not unstable:
            for index in range(frame_count):
                current_target = right_command_target(initial_target, addresses, arm_command[index], hand_command[index])
                max_hand_contacts = 0
                max_finger_count = 0
                last_support_contact = False
                max_arm_contacts = 0
                frame_complete = True
                for step in range(args.substeps):
                    fraction = float(step + 1) / args.substeps
                    target = previous_target + fraction * (current_target - previous_target)
                    set_servo_targets(data, addresses, servos, target)
                    mujoco.mj_step(model, data)
                    hand_count, fingers, support, arm_count = object_contact_diagnostics(model, data, object_geom)
                    max_hand_contacts = max(max_hand_contacts, hand_count)
                    max_finger_count = max(max_finger_count, len(fingers))
                    last_support_contact = support
                    max_arm_contacts = max(max_arm_contacts, arm_count)
                    if has_critical_warning(data):
                        unstable = True
                        frame_complete = False
                        break
                if not frame_complete:
                    break
                positions[index] = data.xpos[object_body]
                quaternions[index] = data.xquat[object_body]
                relative_positions[index] = object_relative_to_body(model, data, flange_body, positions[index])
                arm_actual[index], hand_actual[index] = read_right_state(data, addresses)
                hand_contacts[index] = max_hand_contacts
                finger_counts[index] = max_finger_count
                support_contacts[index] = last_support_contact
                arm_contacts[index] = max_arm_contacts
                executed_frames += 1
                previous_target = current_target
        warnings = warning_counts(data)

    recorded = slice(0, executed_frames)
    phase_recorded = phases[recorded]
    hand_effective_command = np.asarray([
        right_hand_qpos_from_vector(right_command_target(initial_target, addresses, arm_command[index], hand_command[index]), addresses)
        for index in range(len(arm_command))
    ], dtype=np.float64)
    position_recorded = positions[recorded]
    quaternion_recorded = quaternions[recorded]
    relative_recorded = relative_positions[recorded]
    arm_actual_recorded = arm_actual[recorded]
    hand_actual_recorded = hand_actual[recorded]
    hand_contact_recorded = hand_contacts[recorded]
    finger_count_recorded = finger_counts[recorded]
    support_recorded = support_contacts[recorded]
    arm_contact_recorded = arm_contacts[recorded]
    finite_positions = bool(executed_frames > 0 and np.isfinite(position_recorded).all())
    displacement = np.linalg.norm(position_recorded - initial_position[None], axis=1) if finite_positions else np.asarray([np.inf])
    max_displacement = float(np.max(displacement))
    physics_stable = bool(not unstable and executed_frames == len(arm_command) and finite_positions and np.max(np.abs(position_recorded)) <= args.max_absolute_position_m)
    object_left_workspace = bool(max_displacement > args.max_displacement_m)
    arm_tracking_error = float(np.max(np.abs(arm_actual_recorded - arm_command[recorded]))) if executed_frames else float("inf")
    hand_tracking_error = float(np.max(np.abs(hand_actual_recorded - hand_effective_command[recorded]))) if executed_frames else float("inf")
    tracks_command = bool(arm_tracking_error <= args.max_arm_tracking_error_rad and hand_tracking_error <= args.max_hand_tracking_error_rad)
    lift_indices = np.flatnonzero(np.isin(phase_recorded, ("lift", "lift_hold")))
    if len(lift_indices):
        lift_start = max(0, int(lift_indices[0]) - 1)
        lift_delta = float(position_recorded[-1, 2] - position_recorded[lift_start, 2])
        relative_drift = float(np.max(np.linalg.norm(relative_recorded[lift_indices] - relative_recorded[lift_start], axis=1)))
        lift_contact_ratio = float(np.mean(hand_contact_recorded[lift_indices] > 0))
    else:
        lift_delta = float("nan")
        relative_drift = float("inf")
        lift_contact_ratio = 0.0
    close_or_lift = np.isin(phase_recorded, ("close", "close_hold", "lift", "lift_hold"))
    max_finger_count = int(finger_count_recorded[close_or_lift].max()) if np.any(close_or_lift) else 0
    hand_contact_observed = bool(np.any(hand_contact_recorded > 0))
    enough_fingers = bool(max_finger_count >= 2)
    arm_contact_observed = bool(np.any(arm_contact_recorded > 0))
    final_support_contact = bool(support_recorded[-1]) if executed_frames else True
    success = bool(
        physics_stable and tracks_command and not object_left_workspace and hand_contact_observed
        and enough_fingers and not arm_contact_observed and lift_contact_ratio >= 0.5 and not final_support_contact
        and lift_delta >= args.min_lift_m and relative_drift <= args.max_relative_drift_m
    )
    reason = failure_reason(
        physics_stable, tracks_command, hand_contact_observed, enough_fingers, arm_contact_observed, lift_contact_ratio,
        final_support_contact, lift_delta, relative_drift, object_left_workspace,
        args.min_lift_m, args.max_relative_drift_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, timestamps=np.arange(executed_frames, dtype=np.float64) * args.command_time_scale / args.fps, phase=phase_recorded,
        ar5_command=arm_command[recorded].astype(np.float32), ar5_qpos=arm_actual_recorded.astype(np.float32),
        l25_requested_qpos=hand_command[recorded].astype(np.float32), l25_command_qpos=hand_effective_command[recorded].astype(np.float32), l25_qpos=hand_actual_recorded.astype(np.float32),
        object_position=position_recorded.astype(np.float32), object_quaternion=quaternion_recorded.astype(np.float32),
        object_relative_to_flange=relative_recorded.astype(np.float32), object_contact_count=hand_contact_recorded,
        hand_contact_count=hand_contact_recorded, contact_finger_count=finger_count_recorded,
        support_contact=support_recorded, arm_contact_count=arm_contact_recorded,
        mujoco_warning_counts=np.asarray([warnings[name] for name in MUJOCO_WARNING_NAMES.values()], dtype=np.int32),
    )
    report = {
        "frames": int(len(arm_command)), "executed_frames": executed_frames, "phase": phase_recorded.tolist(),
        "T_luban_world_anydex_ar5_base": bridge.tolist(),
        "frame_bridge_translation_variation_m": bridge_translation_variation,
        "frame_bridge_rotation_variation_rad": bridge_rotation_variation,
        "object_initial_body_position_luban_world": object_body_position.tolist(), "object_world_mesh": str(world_mesh_output),
        "object_initial_support_contact": initial_support_contact, "fps": args.fps, "command_time_scale": args.command_time_scale, "substeps": args.substeps,
        "settle_substeps": args.settle_substeps, "mujoco_warning_counts": warnings,
        "max_arm_tracking_error_rad": arm_tracking_error, "max_hand_tracking_error_rad": hand_tracking_error,
        "max_allowed_arm_tracking_error_rad": args.max_arm_tracking_error_rad,
        "max_allowed_hand_tracking_error_rad": args.max_hand_tracking_error_rad,
        "right_hand_contact_observed": hand_contact_observed, "max_right_hand_contact_count": int(hand_contact_recorded.max()) if executed_frames else 0,
        "arm_object_contact_observed": arm_contact_observed, "max_arm_object_contact_count": int(arm_contact_recorded.max()) if executed_frames else 0,
        "max_contact_finger_count": max_finger_count, "lift_hand_contact_ratio": lift_contact_ratio,
        "final_support_contact": final_support_contact, "object_lift_delta_m": lift_delta,
        "min_lift_m": args.min_lift_m, "max_relative_object_drift_m": relative_drift,
        "max_allowed_relative_object_drift_m": args.max_relative_drift_m,
        "max_displacement_m": args.max_displacement_m, "max_observed_displacement_m": max_displacement,
        "physics_stable": physics_stable, "object_left_workspace": object_left_workspace,
        "lift_success": success, "failure_reason": reason,
        "interpretation": "offline MuJoCo contact/lift check with bounded position servos",
        "hardware_command_generated": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
