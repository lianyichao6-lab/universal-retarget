"""跑到指定帧后打开 MuJoCo 交互窗口，手动截图。

用法:
    python test/view_frame.py --config config/shadow_hand.yaml --frame 40
    python test/view_frame.py --config config/wuji_hand.yaml --frame 215
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from qsq_retargeting import Retargeter

# ── landmark 预处理 (与 video.py 一致) ───────────────────────────────
_REFERENCE_WRIST_TO_MIDDLE_MCP = 0.092
_REFERENCE_SEGMENT_LENGTHS = {
    "thumb": [0.0505, 0.0318, 0.0302], "index": [0.0418, 0.0243, 0.0223],
    "middle": [0.0489, 0.0289, 0.0227], "ring": [0.0422, 0.0274, 0.0227],
    "pinky": [0.0343, 0.0195, 0.0201],
}
_FINGER_INDICES = {
    "thumb": [1,2,3,4], "index": [5,6,7,8], "middle": [9,10,11,12],
    "ring": [13,14,15,16], "pinky": [17,18,19,20],
}

def _correct_segment_lengths(kp):
    kp_c = kp.copy()
    for name, idx in _FINGER_INDICES.items():
        ref = _REFERENCE_SEGMENT_LENGTHS[name]
        mcp, pip, dip, tip = idx
        base = kp_c[mcp].copy()
        for i, (a, b, rl) in enumerate([(mcp,pip,ref[0]),(pip,dip,ref[1]),(dip,tip,ref[2])]):
            seg = kp[b] - kp[a]; n = np.linalg.norm(seg)
            if n > 1e-6:
                kp_c[b] = (base if i == 0 else kp_c[a]) + seg / n * rl
    return kp_c

def process_landmarks(kp, depth_scale=1.25):
    kp = kp - kp[0:1]; d = np.linalg.norm(kp[9])
    if d < 1e-6: return kp
    kp = kp * (_REFERENCE_WRIST_TO_MIDDLE_MCP / d)
    kp = _correct_segment_lengths(kp); kp[:, 2] *= depth_scale
    return kp

def landmarks_to_array(hand_landmarks, w, h):
    kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    kp[:, 0] *= w; kp[:, 1] *= h; kp[:, 2] *= w * 2.5
    return kp

# ── 机器人手配置 (与 teleop_sim.py 一致) ─────────────────────────────
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
        "qpos_mapping": [0,1,2,3,8,9,10,11,12,13,14,15,4,5,6,7],
    },
    "inspire_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "inspire_hand" / "inspire_hand_right_mujoco.xml"),
        "qpos_mapping": [8,9,10,11,0,1,2,3,6,7,4,5],
    },
    "ability_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "ability_hand" / "ability_hand_right_mujoco.xml"),
        "qpos_mapping": [8,9,0,1,2,3,6,7,4,5],
    },
    "leap_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "leap_hand" / "leap_hand_right_mujoco.xml"),
        "qpos_mapping": [0,1,2,3,8,9,10,11,12,13,14,15,4,5,6,7],
    },
    "svh_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "schunk_hand" / "schunk_svh_hand_right_mujoco.xml"),
        "qpos_mapping": [0,1,2,3,8,13,14,15,16,9,10,11,12,4,5,6,7,17,18,19],
    },
    "linkerhand_l21": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linkerhand_l21" / f"linkerhand_l21_{side}_mujoco.xml"),
        "qpos_mapping": [0,1,2,3,4,5,9,10,11,6,7,8,12,13,14,15,16],
        "qpos_servo_alpha": 0.2,
    },
    "rohand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "rohand" / f"rohand_{side}_mujoco.xml"),
        "qpos_mapping": [3,4,1,2,0,13,14,11,12,10,18,19,16,17,15,8,9,6,7,5,20,21,23,24,22],
        "qpos_servo_alpha": 0.18,
    },
    "unitree_dex5_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "unitree_dex5_hand" / f"unitree_dex5_hand_{side}_mujoco.xml"),
        "qpos_mapping": [16,17,18,19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "qpos_servo_alpha": 0.2,
    },
}

def map_urdf_to_mujoco_menagerie(qpos):
    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[2]=qpos[17]; ctrl[3]=qpos[18]; ctrl[4]=qpos[19]
    ctrl[5]=qpos[20]; ctrl[6]=qpos[21]
    ctrl[7]=qpos[0]; ctrl[8]=qpos[1]; ctrl[9]=qpos[2]+qpos[3]
    ctrl[10]=qpos[9]; ctrl[11]=qpos[10]; ctrl[12]=qpos[11]+qpos[12]
    ctrl[13]=qpos[13]; ctrl[14]=qpos[14]; ctrl[15]=qpos[15]+qpos[16]
    ctrl[16]=qpos[4]; ctrl[17]=qpos[5]; ctrl[18]=qpos[6]; ctrl[19]=qpos[7]+qpos[8]
    return ctrl

def apply_qpos_to_mujoco(model, data, qpos, hand_cfg):
    if hand_cfg.get("needs_menagerie_mapping"):
        ctrl = map_urdf_to_mujoco_menagerie(qpos)
    elif "qpos_mapping" in hand_cfg:
        ctrl = qpos[hand_cfg["qpos_mapping"]]
    else:
        ctrl = qpos
    ctrl = np.asarray(ctrl, dtype=np.float32)
    alpha = hand_cfg.get("qpos_servo_alpha")
    if alpha is not None:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]; data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif model.nu > 0:
        n = min(len(ctrl), model.nu)
        data.ctrl[:n] = ctrl[:n]
        for _ in range(200): mujoco.mj_step(model, data)
    else:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="跑到指定帧，打开 MuJoCo 窗口截图")
    parser.add_argument("--config", type=str, required=True, help="配置文件 (相对于 example/)")
    parser.add_argument("--frame", type=int, required=True, help="目标帧号")
    parser.add_argument("--video", type=str, default=str(EXAMPLE_DIR / "data" / "right.mp4"))
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--depth-scale", type=float, default=1.25)
    args = parser.parse_args()

    # 加载配置
    config_file = EXAMPLE_DIR / args.config
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    robot_type = config.get("robot", {}).get("type", "shadow_hand")
    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]

    print(f"配置: {args.config}  机器人: {robot_type}  目标帧: {args.frame}")

    # 加载 Retargeter
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)

    # 打开视频，连续跑到目标帧
    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    expected_label = "Left" if args.hand == "right" else "Right"
    last_valid_kp = None
    final_qpos = None

    print(f"连续跑帧 0 ~ {args.frame} ...")
    for fi in range(args.frame + 1):
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = mp_hands.process(rgb)
        chosen = None
        if det.multi_hand_landmarks and det.multi_handedness:
            for hlm, hcls in zip(det.multi_hand_landmarks, det.multi_handedness):
                if hcls.classification[0].label == expected_label:
                    chosen = hlm; break
            if chosen is None: chosen = det.multi_hand_landmarks[0]
        if chosen is not None:
            kp_raw = landmarks_to_array(chosen, w, h)
            last_valid_kp = process_landmarks(kp_raw, args.depth_scale)
        if last_valid_kp is not None:
            qpos, verbose = retargeter.retarget_verbose(last_valid_kp, apply_filter=True)
            final_qpos = qpos

    cap.release()
    mp_hands.close()

    if final_qpos is None:
        print("未检测到手部，退出"); return

    cost = verbose["cost"]
    print(f"帧 {args.frame}: loss={cost:.4f}")

    # 加载 MuJoCo 模型并应用 qpos
    model_path = hand_cfg["model_path"](args.hand)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    apply_qpos_to_mujoco(model, data, final_qpos, hand_cfg)

    # 打开交互窗口
    print("MuJoCo 窗口已打开，调整视角后截图。关闭窗口退出。")
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        viewer.sync()

if __name__ == "__main__":
    main()
