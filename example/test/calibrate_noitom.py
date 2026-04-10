"""Calibrate segment_scaling for Noitom input.

Collects a few seconds of Noitom data while the user holds their hand flat
(fingers fully extended), then computes the ratio between robot FK link
distances and measured human bone distances. Outputs recommended
segment_scaling values and optionally writes them back to the config YAML.

Usage:
    python test/calibrate_noitom.py --robot wuji
    python test/calibrate_noitom.py --robot wuji --write
    python test/calibrate_noitom.py --config config/adaptive/noitom/noitom_wuji_hand.yaml --write
"""

import argparse
import sys
import time
from pathlib import Path

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
from input.noitom import NoitomInput

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _get_link_pos(robot, link_name, offset=None):
    lid = robot.get_link_index(link_name)
    pose = robot.get_link_pose(lid)
    pos = pose[:3, 3].copy()
    if offset is not None:
        pos += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
    return pos


def get_robot_distances(optimizer):
    """Compute wrist→PIP, wrist→DIP, wrist→TIP distances at neutral/zero pose.

    Returns three lists of length num_fingers, each entry is distance in meters
    (or None if the link is unavailable).
    """
    robot = optimizer.robot
    nf = optimizer.num_fingers

    qpos = optimizer.neutral_qpos.copy() if optimizer.neutral_qpos is not None \
        else np.zeros(robot.model.nq)
    robot.compute_forward_kinematics(qpos)

    origin_pos = _get_link_pos(robot, optimizer.origin_link_name)

    def dists_for(link_names, offsets):
        result = []
        for fi in range(nf):
            if fi >= len(link_names):
                result.append(None)
                continue
            off = offsets[fi] if offsets is not None else None
            pos = _get_link_pos(robot, link_names[fi], off)
            result.append(float(np.linalg.norm(pos - origin_pos)))
        return result

    pip_dists = dists_for(
        getattr(optimizer, 'link3_names', []),
        getattr(optimizer, 'link3_offsets', None),
    )
    dip_dists = dists_for(
        getattr(optimizer, 'link4_names', []),
        getattr(optimizer, 'link4_offsets', None),
    )
    tip_dists = dists_for(
        optimizer.task_link_names,
        getattr(optimizer, 'task_offsets', None),
    )
    return pip_dists, dip_dists, tip_dists


def collect_human_distances(input_device, retargeter, hand, duration):
    """Collect Noitom data for `duration` seconds and compute average
    wrist→PIP, wrist→DIP, wrist→TIP distances per finger."""
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    pip_idx = [optimizer.MP_PIP_INDICES[i] for i in optimizer.mp_finger_indices]
    dip_idx = [optimizer.MP_DIP_INDICES[i] for i in optimizer.mp_finger_indices]
    tip_idx = [optimizer.MP_TIP_INDICES[i] for i in optimizer.mp_finger_indices]

    pip_buf = [[] for _ in range(nf)]
    dip_buf = [[] for _ in range(nf)]
    tip_buf = [[] for _ in range(nf)]

    start = time.time()
    frames = 0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            time.sleep(0.01)
            continue

        kp = apply_mediapipe_transformations(raw_kp, hand)
        if retargeter.rotation_xyz:
            kp = retargeter._apply_rotation(kp)

        wrist = kp[0]
        for i in range(nf):
            pip_buf[i].append(float(np.linalg.norm(kp[pip_idx[i]] - wrist)))
            dip_buf[i].append(float(np.linalg.norm(kp[dip_idx[i]] - wrist)))
            tip_buf[i].append(float(np.linalg.norm(kp[tip_idx[i]] - wrist)))
        frames += 1
        time.sleep(0.01)

    if frames == 0:
        return frames, None, None, None

    pip_mean = [float(np.mean(b)) if b else 0.0 for b in pip_buf]
    dip_mean = [float(np.mean(b)) if b else 0.0 for b in dip_buf]
    tip_mean = [float(np.mean(b)) if b else 0.0 for b in tip_buf]
    return frames, pip_mean, dip_mean, tip_mean


def main():
    parser = argparse.ArgumentParser(description="Calibrate Noitom segment_scaling")
    parser.add_argument("--config", default=None,
                        help="Config YAML path (overrides --robot)")
    parser.add_argument("--robot", default="wuji",
                        choices=["shadow", "wuji", "allegro", "leap", "inspire",
                                 "ability", "svh", "rohand", "linkerhand_l21", "unitree_dex5"])
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--noitom-local-ip", default="0.0.0.0")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="采集时长（秒），默认 5")
    parser.add_argument("--write", action="store_true",
                        help="将结果直接写回配置文件")
    args = parser.parse_args()

    robot_name_map = {
        "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
        "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
        "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
        "unitree_dex5": "unitree_dex5_hand",
    }
    robot_file = robot_name_map.get(args.robot, args.robot)
    config_path = args.config if args.config \
        else f"config/adaptive/noitom/noitom_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path

    print(f"配置文件: {config_file}")
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    fi_names = [FINGER_NAMES[i] for i in optimizer.mp_finger_indices]

    # ── Robot distances ──────────────────────────────────────────────────────
    print("正在计算机器人关节距离（中性位姿）...")
    pip_robot, dip_robot, tip_robot = get_robot_distances(optimizer)

    print(f"\n机器人各关节到腕部距离:")
    print(f"  {'手指':8s}  {'PIP':>7s}  {'DIP':>7s}  {'TIP':>7s}")
    for i in range(nf):
        fmt = lambda v: f"{v*100:.1f}cm" if v else "  N/A"
        print(f"  {fi_names[i]:8s}  {fmt(pip_robot[i]):>7s}  "
              f"{fmt(dip_robot[i]):>7s}  {fmt(tip_robot[i]):>7s}")

    # ── Connect Noitom ───────────────────────────────────────────────────────
    print(f"\n连接诺亦腾 ({args.noitom_local_ip}:{args.noitom_local_port})...")
    input_device = NoitomInput(
        local_ip=args.noitom_local_ip,
        local_port=args.noitom_local_port,
        server_ip=args.noitom_server_ip,
        server_port=args.noitom_server_port,
    )

    # ── Countdown ────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("请伸直所有手指（手掌朝下，五指张开、伸直）")
    print("保持该姿势，3 秒倒计时后开始采集...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print(f"开始采集 {args.duration:.0f} 秒...")

    # ── Collect ──────────────────────────────────────────────────────────────
    frames, pip_human, dip_human, tip_human = collect_human_distances(
        input_device, retargeter, args.hand, args.duration
    )

    if frames == 0:
        print("未收到数据，请检查网络连接。")
        return

    print(f"采集完成，共 {frames} 帧\n")
    print(f"人手各关节到腕部距离（变换后平均值）:")
    print(f"  {'手指':8s}  {'PIP':>7s}  {'DIP':>7s}  {'TIP':>7s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {pip_human[i]*100:>6.1f}cm  "
              f"{dip_human[i]*100:>6.1f}cm  {tip_human[i]*100:>6.1f}cm")

    # ── Compute scaling ──────────────────────────────────────────────────────
    def ratio(robot_d, human_d):
        if robot_d and human_d and human_d > 1e-4:
            return round(robot_d / human_d, 3)
        return 1.0

    result = {}
    for i, fname in enumerate(fi_names):
        result[fname] = [
            ratio(pip_robot[i], pip_human[i]),
            ratio(dip_robot[i], dip_human[i]),
            ratio(tip_robot[i], tip_human[i]),
        ]

    print(f"\n推荐 segment_scaling:")
    print(f"  {'手指':8s}  {'PIP':>6s}  {'DIP':>6s}  {'TIP':>6s}")
    for fname, vals in result.items():
        print(f"  {fname:8s}  {vals[0]:>6.3f}  {vals[1]:>6.3f}  {vals[2]:>6.3f}")

    print("\n复制以下内容到配置文件的 segment_scaling 部分:")
    print("  segment_scaling:")
    for fname, vals in result.items():
        print(f"    {fname}: {vals}")

    # ── Write back ───────────────────────────────────────────────────────────
    if args.write:
        with open(config_file, 'r') as f:
            raw = f.read()
        config = yaml.safe_load(raw)
        config['retarget']['segment_scaling'] = result
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=None, allow_unicode=True,
                      sort_keys=False)
        print(f"\n已写入 {config_file}")


if __name__ == "__main__":
    main()
