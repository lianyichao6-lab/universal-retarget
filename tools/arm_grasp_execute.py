#!/usr/bin/env python3
"""Drive the right AR5 arm to a grasp pose and close the G20 hand.

Phases (--stage):
  preview  — compute and print the arm target (default, no motion)
  open     — open hand (qpos=0)
  arm      — move arm: pregrasp approach -> grasp position
  close    — close hand to grasp qpos
  all      — open hand -> move arm -> close hand (full sequence)

Safety:
  Hardware motion requires both --execute and --confirm ARM_G20_CLEAR.

Usage:
    # Preview from grasp_execution_plan.npz
    python tools/arm_grasp_execute.py \\
      --grasp-plan outputs/.../grasp_execution_plan.npz \\
      --flange-hand /path/to/T_arm_flange_l25_hand.npy

    # Preview from quick_grasp output directory
    python tools/arm_grasp_execute.py \\
      --grasp-dir outputs/.../quick_grasp \\
      --flange-hand /path/to/T_arm_flange_l25_hand.npy

    # Full execution (arm + hand)
    python tools/arm_grasp_execute.py \\
      --grasp-dir outputs/.../quick_grasp \\
      --flange-hand /path/to/T_arm_flange_l25_hand.npy \\
      --arm-ip 192.168.1.18 \\
      --stage all --execute --confirm ARM_G20_CLEAR

Supported inputs (--grasp-plan or --grasp-dir):
  - grasp_execution_plan.npz  (contains T_anchor_l25_hand + l25_qpos)
  - l25_collision_aware_plan.npz  (contains camera_to_l25_rotation/translation + qpos)
  - quick_grasp output directory  (contains trajectory.npz + canonical_grasp.npz)

Prerequisites:
  - xCoreSDK (set XCORESDK_PY_PATH)
  - /botclaw/calib_results/ with full_calibration_result.npz + T_leftbase_rightbase.npy
  - CAN0 UP + LINKERHAND_SDK_PACKAGE for hand control
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SDK_PY_PATH = os.environ.get(
    "XCORESDK_PY_PATH",
    str(
        Path.home()
        / "luban_framework"
        / "src"
        / "perception"
        / "Arms_SDK_v0.7.1_py"
        / "Release"
        / "linux"
    ),
)
if os.path.isdir(_SDK_PY_PATH):
    sys.path.insert(0, _SDK_PY_PATH)

DEFAULT_CALIB_DIR = "/botclaw/calib_results"
CONFIRMATION_TOKEN = "ARM_G20_CLEAR"
VECTOR_CONFIG = (
    ROOT / "example" / "config" / "vector" / "mediapipe"
    / "mediapipe_linkerhand_l25.yaml"
)


def _similarity(
    source: np.ndarray, target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation for target = scale * source @ R.T + t."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Degenerate hand keypoints")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


# ── Calibration ──────────────────────────────────────────


def _load_calibration(calib_dir: Path) -> dict[str, np.ndarray]:
    full_path = calib_dir / "full_calibration_result.npz"
    lr_path = calib_dir / "T_leftbase_rightbase.npy"
    if not full_path.exists():
        raise FileNotFoundError(f"标定文件不存在: {full_path}")
    if not lr_path.exists():
        raise FileNotFoundError(f"标定文件不存在: {lr_path}")

    full = np.load(str(full_path), allow_pickle=True)
    T_world_headcam = full["T_world_headcam"].astype(np.float64)
    T_world_leftbase = full["T_world_leftbase"].astype(np.float64)
    T_leftbase_rightbase = np.load(str(lr_path)).astype(np.float64)

    T_world_rightbase = T_world_leftbase @ T_leftbase_rightbase
    # anchor = headcam (zed_left_camera_frame_optical)
    T_robot_base_anchor = np.linalg.inv(T_world_rightbase) @ T_world_headcam

    return {
        "T_world_headcam": T_world_headcam,
        "T_world_rightbase": T_world_rightbase,
        "T_robot_base_anchor": T_robot_base_anchor,
    }


def _load_flange_hand(path_str: str) -> np.ndarray:
    if path_str.lower() == "identity":
        print("  [注意] 使用单位矩阵作为 T_arm_flange_l25_hand (仅测试)")
        return np.eye(4, dtype=np.float64)
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"T_arm_flange_l25_hand 文件不存在: {path}")
    if path.suffix == ".npy":
        return np.load(str(path), allow_pickle=False).astype(np.float64)
    with np.load(str(path), allow_pickle=False) as data:
        for key in ("T_arm_flange_l25_hand", "T_flange_hand"):
            if key in data:
                return data[key].astype(np.float64)
    raise ValueError(f"未找到 T_arm_flange_l25_hand in {path}")


def _load_grasp_plan(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as data:
        plan = {k: np.asarray(data[k]).copy() for k in data.files}

    if "T_anchor_l25_hand" in plan:
        return {
            "T_anchor_l25_hand": plan["T_anchor_l25_hand"].astype(np.float64),
            "l25_qpos": np.asarray(plan["l25_qpos"], dtype=np.float64),
            "anchor_frame": str(plan.get("anchor_frame", np.asarray("")).item()),
            "candidate_id": str(
                plan.get("candidate_id", np.asarray("")).item()
            ),
        }

    if "camera_to_l25_rotation" in plan:
        from anydexretarget.deployment import rigid_transform

        T_l25_anchor = rigid_transform(
            plan["camera_to_l25_rotation"],
            plan["camera_to_l25_translation"],
        )
        return {
            "T_anchor_l25_hand": np.linalg.inv(T_l25_anchor),
            "l25_qpos": np.asarray(plan["qpos"], dtype=np.float64),
            "anchor_frame": "zed_left_camera_frame_optical",
            "candidate_id": "",
        }

    raise ValueError(
        f"无法识别的手势文件格式: {path}\n"
        "需包含 T_anchor_l25_hand 或 camera_to_l25_rotation"
    )


def _load_grasp_from_dir(grasp_dir: Path) -> tuple[dict, Path | None]:
    """Load grasp from quick_grasp output directory.

    Computes the camera-to-L25 similarity transform by re-running the
    retargeter on the canonical keypoints (same method as
    plan_l25_object_relative_grasp.py lines 117-118).

    Returns (grasp_dict, trajectory_pkl_path).
    """
    traj_npz = grasp_dir / "trajectory.npz"
    canon_npz = grasp_dir / "canonical_grasp.npz"
    traj_pkl = grasp_dir / "trajectory.pkl"
    if not traj_npz.exists():
        raise FileNotFoundError(f"未找到 {traj_npz}")
    if not canon_npz.exists():
        raise FileNotFoundError(f"未找到 {canon_npz}")

    with np.load(str(traj_npz), allow_pickle=False) as data:
        robot_qpos = np.asarray(data["robot_qpos"], dtype=np.float64)
    l25_qpos = robot_qpos[-1]

    print("  计算 camera→L25 相似变换 (retarget FK)...")
    from anydexretarget.hand_representation import load_canonical_grasp_state
    from anydexretarget.retarget import Retargeter

    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    state = load_canonical_grasp_state(canon_npz)
    source_hand = state.keypoints_for_retargeting().astype(np.float64)
    _, verbose = retargeter.retarget_verbose(
        source_hand, apply_filter=False
    )
    transformed_hand = np.asarray(
        verbose["mediapipe_kp"], dtype=np.float64
    )
    scale, rotation, translation = _similarity(
        source_hand, transformed_hand
    )
    print(f"  scale={scale:.4f}, |t|={np.linalg.norm(translation):.4f} m")

    from anydexretarget.deployment import rigid_transform

    T_l25_anchor = rigid_transform(rotation, translation)
    T_anchor_l25_hand = np.linalg.inv(T_l25_anchor)

    grasp = {
        "T_anchor_l25_hand": T_anchor_l25_hand,
        "l25_qpos": l25_qpos,
        "anchor_frame": "zed_left_camera_frame_optical",
        "candidate_id": "",
    }
    found_pkl = traj_pkl if traj_pkl.exists() else None
    return grasp, found_pkl


# ── Arm SDK helpers (参考 verify_tcp_tool_calibration.py) ─


def _import_arm_sdk():
    from xCoreSDK_python import (
        Cobot_7,
        CoordinateType,
        MoveAbsJCommand,
        MotionControlMode,
        OperateMode,
        OperationState,
        PowerState,
        PyString,
    )

    return (
        Cobot_7,
        CoordinateType,
        MoveAbsJCommand,
        MotionControlMode,
        OperateMode,
        OperationState,
        PowerState,
        PyString,
    )


def _connect_arm(ip: str, local_ip: str, speed: float):
    (
        Cobot_7,
        _,
        _,
        MotionControlMode,
        OperateMode,
        _,
        PowerState,
        _,
    ) = _import_arm_sdk()
    if not local_ip:
        print(f"  [警告] 未指定 --arm-local-ip, 运动命令可能无法正常下发!")
        print(f"         建议使用 --arm-local-ip 192.168.2.100")
    print(f"  连接机械臂 {ip} (local={local_ip or 'N/A'})...")
    arm = Cobot_7(ip, local_ip) if local_ip else Cobot_7(ip)
    ec = {}
    arm.setSimulationMode(False, ec)
    ec = {}
    arm.clearServoAlarm(ec)
    ec = {}
    arm.setOperateMode(OperateMode.automatic, ec)
    time.sleep(0.3)
    arm.setPowerState(True, ec)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if arm.powerState(ec) == PowerState.on:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"机械臂 {ip} 上电超时")
    time.sleep(1)
    arm.setMotionControlMode(MotionControlMode.NrtCommandMode, ec)
    arm.setDefaultSpeed(speed, ec)
    print(f"  机械臂已连接 (速度={speed}mm/s)")
    return arm


def _get_arm_joints(arm) -> list:
    ec = {}
    return list(arm.jointPos(ec))[:7]


def _get_flange_pose(arm) -> np.ndarray:
    (_, CoordinateType, *_) = _import_arm_sdk()
    ec = {}
    return np.array(list(arm.posture(CoordinateType.flangeInBase, ec)))


class _SDKIKClient:
    def __init__(self, arm):
        self._model = arm.model()
        identity16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self._model.setTcpCoor(identity16, identity16)
        self._joint_limits = self._read_soft_limits(arm)
        if self._joint_limits:
            lim_str = ", ".join(
                f"J{i}[{lo*180/math.pi:.0f},{hi*180/math.pi:.0f}]"
                for i, (lo, hi) in enumerate(self._joint_limits)
            )
            print(f"  关节软限位(deg): {lim_str}")

    @staticmethod
    def _read_soft_limits(arm):
        from xCoreSDK_python import PyTypeVectorArrayDouble2

        lim = PyTypeVectorArrayDouble2()
        ec = {}
        if arm.getSoftLimit(lim, ec):
            pairs = lim.content()
            if pairs and len(pairs) >= 7:
                return [(lo, hi) for lo, hi in pairs[:7]]
        return None

    def _within_limits(self, joints: list) -> bool:
        if self._joint_limits is None:
            return True
        for i, (lo, hi) in enumerate(self._joint_limits):
            if joints[i] < lo or joints[i] > hi:
                return False
        return True

    def forward(self, joint_positions):
        mat16 = self._model.getCartPose(list(joint_positions))
        if mat16 is None or len(mat16) < 16:
            return None
        T = np.eye(4)
        T[:3, :3] = np.array(
            [
                [mat16[0], mat16[1], mat16[2]],
                [mat16[4], mat16[5], mat16[6]],
                [mat16[8], mat16[9], mat16[10]],
            ]
        )
        T[:3, 3] = [mat16[3], mat16[7], mat16[11]]
        return T

    def solve_flange_mat16(self, flange_mat16: list, seed_joints: list):
        best_joints = None
        best_dist = float("inf")
        for psi_deg in range(-180, 180, 5):
            psi = math.radians(psi_deg)
            output = []
            ret = self._model.getJointPos(
                flange_mat16, psi, list(seed_joints), output
            )
            if ret == 0 and output and len(output) >= 7:
                candidate = list(output)[:7]
                if not self._within_limits(candidate):
                    continue
                dist = sum(
                    (candidate[i] - seed_joints[i]) ** 2 for i in range(7)
                )
                if dist < best_dist:
                    best_dist = dist
                    best_joints = candidate
        return best_joints


def _mat4_to_mat16(T: np.ndarray) -> list:
    return [
        T[0, 0], T[0, 1], T[0, 2], T[0, 3],
        T[1, 0], T[1, 1], T[1, 2], T[1, 3],
        T[2, 0], T[2, 1], T[2, 2], T[2, 3],
        0.0, 0.0, 0.0, 1.0,
    ]


def _move_arm_to_flange(
    arm, T_flange: np.ndarray, ik: _SDKIKClient, speed: float
):
    from xCoreSDK_python import (
        Event,
        MoveAbsJCommand,
        OperationState,
        PyString,
    )

    mat16 = _mat4_to_mat16(T_flange)
    seed = _get_arm_joints(arm)
    joints = ik.solve_flange_mat16(mat16, seed)
    if joints is None:
        pos = T_flange[:3, 3] * 1000
        limits_msg = ""
        if ik._joint_limits:
            limits_msg = "\n    关节限位(deg): " + ", ".join(
                f"J{i}[{lo*180/math.pi:.0f},{hi*180/math.pi:.0f}]"
                for i, (lo, hi) in enumerate(ik._joint_limits)
            )
        raise RuntimeError(
            f"IK 求解失败 (所有 psi 解均超关节限位或无解)\n"
            f"    目标位置: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm"
            f"{limits_msg}"
        )

    joint_diff = [abs(joints[i] - seed[i]) * 180 / math.pi for i in range(7)]
    print(
        f"    IK: 当前关节(deg)={[f'{j*180/math.pi:.1f}' for j in seed]}"
    )
    print(
        f"    IK: 目标关节(deg)={[f'{j*180/math.pi:.1f}' for j in joints]}"
    )
    print(
        f"    IK: 关节差(deg)={[f'{d:.1f}' for d in joint_diff]}, "
        f"max={max(joint_diff):.1f}°"
    )

    # Verify IK solution via FK
    fk_result = ik.forward(joints)
    if fk_result is not None:
        fk_pos = fk_result[:3, 3] * 1000
        target_pos = T_flange[:3, 3] * 1000
        fk_err = np.linalg.norm(fk_result[:3, 3] - T_flange[:3, 3]) * 1000
        print(
            f"    FK 验证: IK解→位置=[{fk_pos[0]:.1f}, {fk_pos[1]:.1f}, {fk_pos[2]:.1f}] mm, "
            f"目标=[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}] mm, "
            f"误差={fk_err:.2f} mm"
        )

    ec = {}
    cur_state = arm.operationState(ec)
    print(f"    运动前状态: {cur_state}")
    if cur_state == OperationState.moving:
        arm.stop(ec)
        time.sleep(0.5)

    arm.moveReset(ec)
    print(f"    moveReset ec={ec}")

    cmd_id = PyString()
    ec = {}
    arm.moveAppend(
        [MoveAbsJCommand(joints, speed, 0.0)], cmd_id, ec
    )
    print(f"    moveAppend ec={ec}, cmd_id={cmd_id}")
    if ec.get("ec", -1) != 0:
        raise RuntimeError(f"moveAppend 失败: {ec}")

    for attempt in range(3):
        ec = {}
        arm.moveStart(ec)
        print(f"    moveStart attempt={attempt} ec={ec}")
        if ec.get("ec", -1) == 0:
            break
        if attempt < 2:
            time.sleep(1)
            arm.moveReset(ec)
            ec_tmp = {}
            cmd_id = PyString()
            arm.moveAppend(
                [MoveAbsJCommand(joints, speed, 0.0)], cmd_id, ec_tmp
            )
    else:
        raise RuntimeError(f"moveStart 失败 (重试3次): {ec}")

    time.sleep(0.3)
    state_after_start = arm.operationState({})
    print(f"    moveStart 后状态: {state_after_start}")

    if state_after_start != OperationState.moving:
        time.sleep(1.0)
        state_retry = arm.operationState({})
        joints_after = _get_arm_joints(arm)
        joint_moved = max(
            abs(joints_after[i] - seed[i]) * 180 / math.pi
            for i in range(7)
        )
        if joint_moved < 0.01:
            ev_ec = {}
            ev = arm.queryEventInfo(Event.moveExecution, ev_ec)
            err_msg = ev.get("error", {}).get("message", "unknown")
            err_code = ev.get("error", {}).get("ec", "?")
            raise RuntimeError(
                f"臂未运动! moveExecution error: [{err_code}] {err_msg}"
            )

    deadline = time.time() + 45.0
    while time.time() < deadline:
        ec_poll = {}
        state = arm.operationState(ec_poll)
        if state != OperationState.moving:
            print(f"    运动结束, 状态={state}")
            return
        time.sleep(0.1)
    raise TimeoutError("MoveAbsJ 超时 (45s)")


# ── Hand execution ───────────────────────────────────────


def _execute_hand(
    l25_qpos: np.ndarray,
    trajectory_pkl: Path | None,
    hand_frame: int,
    args: argparse.Namespace,
):
    sdk_pkg = args.sdk_package or os.environ.get(
        "LINKERHAND_SDK_PACKAGE", ""
    )
    if not sdk_pkg:
        raise RuntimeError(
            "需设置 --sdk-package 或 LINKERHAND_SDK_PACKAGE 环境变量"
        )

    cleanup = False
    if trajectory_pkl is None:
        records = [{"target": l25_qpos.copy(), "frame": 0}]
        fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="grasp_traj_")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            pickle.dump(records, f)
        trajectory_pkl = Path(tmp_path)
        hand_frame = 0
        cleanup = True

    cmd = [
        "/usr/bin/python3",
        str(ROOT / "tools" / "g20_hardware_execute.py"),
        "--sdk-package", sdk_pkg,
        "--trajectory", str(trajectory_pkl),
        "--frame", str(hand_frame),
        "--hardware",
        "--confirm", "G20_RIGHT_CLEAR",
        "--read-state",
        "--channels",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        "--replay-target",
        "--rate-hz", str(args.rate_hz),
        "--max-step", str(args.max_step),
        "--speed", str(args.hand_speed),
        "--torque", str(args.hand_torque),
    ]
    try:
        print(f"  执行手部控制...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"G20 手部执行返回 exit code {result.returncode}"
            )
    finally:
        if cleanup:
            os.unlink(str(trajectory_pkl))


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="驱动右臂到手势位置，然后抓取物体",
    )
    parser.add_argument(
        "--grasp-plan",
        type=Path,
        default=None,
        help="手势文件 (.npz): grasp_execution_plan 或 l25_collision_aware_plan",
    )
    parser.add_argument(
        "--grasp-dir",
        type=Path,
        default=None,
        help="quick_grasp 输出目录 (含 trajectory.npz + canonical_grasp.npz)",
    )
    parser.add_argument(
        "--flange-hand",
        required=True,
        help="T_arm_flange_l25_hand 标定文件 (.npy/.npz) 或 'identity'",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="L25 手部轨迹 (.pkl); 不提供则从 l25_qpos 生成单帧目标",
    )
    parser.add_argument(
        "--hand-frame",
        type=int,
        default=0,
        help="--trajectory 中使用的帧索引 (default: 0)",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path(DEFAULT_CALIB_DIR),
    )
    parser.add_argument("--arm-ip", default="", help="右臂 IP (执行时必需)")
    parser.add_argument(
        "--arm-local-ip",
        default="192.168.2.100",
        help="本机 IP (default: 192.168.2.100, 与标定一致)",
    )
    parser.add_argument(
        "--arm-speed",
        type=float,
        default=50.0,
        help="臂运动速度 mm/s (default: 50)",
    )
    parser.add_argument(
        "--grasp-offset",
        type=str,
        default="0,0,0",
        help="抓取位置微调 (mm, 臂base坐标系 x,y,z), 如 '0,-10,0'",
    )
    parser.add_argument(
        "--lift-offset",
        type=str,
        default="100,200,0",
        help="抓取后抬起偏移 (mm, 臂base坐标系 x,y,z), 如 '100,200,0' = +x 10cm, +y 20cm",
    )
    parser.add_argument(
        "--stage",
        choices=("preview", "open", "arm", "close", "hand", "all"),
        default="preview",
        help="preview=仅计算, open=张开手, arm=移臂, close/hand=闭合手, all=张开→移臂→闭合",
    )
    parser.add_argument(
        "--sdk-package",
        default="",
        help="LINKERHAND_SDK_PACKAGE 路径 (手部控制)",
    )
    parser.add_argument("--max-step", type=int, default=5)
    parser.add_argument("--rate-hz", type=int, default=10)
    parser.add_argument("--hand-speed", type=int, default=30)
    parser.add_argument("--hand-torque", type=int, default=40)
    parser.add_argument(
        "--execute", action="store_true", help="启用硬件运动"
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"安全确认令牌: {CONFIRMATION_TOKEN}",
    )
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRMATION_TOKEN:
        parser.error(f"硬件执行需要 --confirm {CONFIRMATION_TOKEN}")
    if args.stage in ("arm", "all") and args.execute and not args.arm_ip:
        parser.error("臂运动需要 --arm-ip")
    if not args.grasp_plan and not args.grasp_dir:
        parser.error("需要 --grasp-plan 或 --grasp-dir")
    if args.grasp_plan and args.grasp_dir:
        parser.error("--grasp-plan 和 --grasp-dir 不能同时使用")

    # ── 1. Load grasp plan ──
    print("=" * 60)
    print(" 手臂抓取执行")
    print("=" * 60)

    auto_trajectory: Path | None = None
    print("\n[1/4] 加载手势文件...")
    if args.grasp_dir:
        print(f"  目录: {args.grasp_dir}")
        grasp, auto_trajectory = _load_grasp_from_dir(args.grasp_dir)
    else:
        print(f"  文件: {args.grasp_plan}")
        grasp = _load_grasp_plan(args.grasp_plan)
        auto_trajectory = None

    T_anchor_hand = grasp["T_anchor_l25_hand"]
    l25_qpos = grasp["l25_qpos"]
    print(f"  anchor_frame: {grasp['anchor_frame']}")
    if grasp["candidate_id"]:
        print(f"  candidate: {grasp['candidate_id']}")
    print(
        f"  l25_qpos 范围: [{l25_qpos.min():.3f}, {l25_qpos.max():.3f}] rad"
    )

    # ── 2. Compute arm flange target ──
    print("\n[2/4] 计算臂目标位姿...")
    print(f"  标定目录: {args.calib_dir}")
    calib = _load_calibration(args.calib_dir)
    T_flange_hand = _load_flange_hand(args.flange_hand)

    T_robot_base_anchor = calib["T_robot_base_anchor"]

    # T_robot_base_arm_flange = T_robot_base_anchor @ T_anchor_l25_hand @ inv(T_arm_flange_l25_hand)
    T_flange_target = (
        T_robot_base_anchor @ T_anchor_hand @ np.linalg.inv(T_flange_hand)
    )

    grasp_offset_mm = [float(x) for x in args.grasp_offset.split(",")]
    if len(grasp_offset_mm) != 3:
        parser.error("--grasp-offset 需要 3 个值, 如 '0,-10,0'")
    grasp_offset_m = np.array(grasp_offset_mm) / 1000.0
    if np.any(grasp_offset_m != 0):
        T_flange_target[:3, 3] += grasp_offset_m
        print(
            f"  抓取微调 (base frame): "
            f"[{grasp_offset_mm[0]:.1f}, {grasp_offset_mm[1]:.1f}, {grasp_offset_mm[2]:.1f}] mm"
        )

    from anydexretarget.luban_arm import arm_flange_pose_xyzw

    position, quaternion = arm_flange_pose_xyzw(T_flange_target)
    euler = Rot.from_matrix(T_flange_target[:3, :3]).as_euler(
        "xyz", degrees=True
    )

    print(
        f"  执行公式: T_base_flange = T_base_anchor @ T_anchor_hand @ inv(T_flange_hand)"
    )
    print(
        f"  目标 flange 位置 (r_base_link): "
        f"[{position[0]*1000:.1f}, {position[1]*1000:.1f}, {position[2]*1000:.1f}] mm"
    )
    print(
        f"  目标 flange 四元数 (xyzw): "
        f"[{quaternion[0]:.4f}, {quaternion[1]:.4f}, "
        f"{quaternion[2]:.4f}, {quaternion[3]:.4f}]"
    )
    print(
        f"  目标 flange 欧拉角: "
        f"[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg"
    )

    if not args.execute:
        print(f"\n  预览模式 — 无硬件运动")
        print(
            f"  添加 --execute --confirm {CONFIRMATION_TOKEN} 以执行"
        )
        return

    # ── 3. Open hand ──
    arm = None
    ik = None
    if args.stage in ("all",):
        print(f"\n[3/6] 张开 G20 手 (初始化)...")
        open_qpos = np.zeros_like(l25_qpos)
        _execute_hand(open_qpos, None, 0, args)
        print(f"  手已张开")
    elif args.stage == "open":
        print(f"\n[3/6] 张开 G20 手...")
        open_qpos = np.zeros_like(l25_qpos)
        _execute_hand(open_qpos, None, 0, args)
        print(f"  手已张开")
    else:
        print(f"\n[3/6] 跳过手部张开")

    # ── 4. Move arm ──
    if args.stage in ("arm", "all"):
        print(f"\n[4/6] 驱动右臂...")
        arm = _connect_arm(args.arm_ip, args.arm_local_ip, args.arm_speed)
        ik = _SDKIKClient(arm)

        cur_pose = _get_flange_pose(arm)
        print(
            f"  当前 flange 位置: "
            f"[{cur_pose[0]*1000:.1f}, {cur_pose[1]*1000:.1f}, {cur_pose[2]*1000:.1f}] mm"
        )

        try:
            print(f"  -> 移动到抓取位置...")
            _move_arm_to_flange(
                arm, T_flange_target, ik, args.arm_speed
            )

            after_pose = _get_flange_pose(arm)
            error = np.linalg.norm(
                after_pose[:3] - T_flange_target[:3, 3]
            )
            print(
                f"  臂到位 (定位误差: {error*1000:.2f} mm)"
            )
        except Exception as e:
            print(f"  [错误] 臂运动失败: {e}")
            if arm is not None:
                arm.disconnectFromRobot({})
            raise
    else:
        print(f"\n[4/6] 跳过臂运动 (--stage {args.stage})")

    # ── 5. Close hand (grasp) ──
    if args.stage in ("hand", "close", "all"):
        print(f"\n[5/6] 闭合 G20 手 (抓取)...")
        traj = args.trajectory or auto_trajectory
        _execute_hand(l25_qpos, traj, args.hand_frame, args)
        print(f"  抓取完成")
    else:
        print(f"\n[5/6] 跳过手部闭合 (--stage {args.stage})")

    # ── 6. Lift after grasp ──
    lift_mm = [float(x) for x in args.lift_offset.split(",")]
    if len(lift_mm) != 3:
        parser.error("--lift-offset 需要 3 个值, 如 '100,200,0'")
    lift_m = np.array(lift_mm) / 1000.0
    has_lift = np.any(lift_m != 0)

    if args.stage == "all" and has_lift and arm is not None:
        print(
            f"\n[6/6] 抬起 "
            f"[{lift_mm[0]:.0f}, {lift_mm[1]:.0f}, {lift_mm[2]:.0f}] mm..."
        )
        T_lift = T_flange_target.copy()
        T_lift[:3, 3] += lift_m
        try:
            _move_arm_to_flange(arm, T_lift, ik, args.arm_speed)
            lift_pose = _get_flange_pose(arm)
            lift_err = np.linalg.norm(lift_pose[:3] - T_lift[:3, 3]) * 1000
            print(f"  抬起到位 (误差: {lift_err:.2f} mm)")
        except Exception as e:
            print(f"  [错误] 抬起失败: {e}")
    else:
        print(f"\n[6/6] 跳过抬起")

    if arm is not None:
        arm.disconnectFromRobot({})

    print("\n" + "=" * 60)
    print(" 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
