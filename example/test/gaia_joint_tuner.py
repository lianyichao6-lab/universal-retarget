"""GaiaHand20 关节调试工具：拖动 MuJoCo Control 面板滑条，同步控制仿真和真机。

用途：
  交互式调整每个关节角——拖滑条观察仿真姿态，加 --real 时真机同步跟随；
  按 P 打印当前所有关节角（弧度），方便记录标定数值。

用法：
  # 仅仿真（不连真机）
  python example/test/gaia_joint_tuner.py --hand right

  # 仿真 + 同步发送到真机
  python example/test/gaia_joint_tuner.py --hand right --real --gaia-port /dev/ttyACM0

操作方式：
  - 打开 MuJoCo 右侧 Joint → Control 面板，拖动任意滑条
  - 仿真立即更新；加 --real 时真机同步跟随（约 20 Hz）
  - P     打印当前关节角 JSON

参数：
  --hand          手的左右侧，right 或 left（默认 right）
  --real          同时发送到真机
  --gaia-port     Gaia 串口，默认 /dev/ttyACM0
  --gaia-baudrate 波特率，默认 921600
  --no-slcan      禁用 SLCAN 模式（默认启用）
  --no-main-board 声明无主控板（默认有主控板）
  --speed         Gaia 运动速度 (0, 1]，默认 0.7
  --lpf-level     Gaia 电机 LPF 等级 0-5，默认 3
  --pip-scale     四指 joint_3 (PIP) 发送给真机的缩放系数，默认 1.0
                  真机弯曲不足时调大（如 1.2），超过关节上限自动截断
  --scene         手动指定 MuJoCo 场景 XML 路径
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Finger joint counts for the Gaia SDK (active joints only).
# thumb: 4 joints, index/middle/ring/little: 3 joints each = 16 total.
_GAIA_FINGER_JOINT_COUNTS = (
    ("thumb",  4),
    ("index",  3),
    ("middle", 3),
    ("ring",   3),
    ("little", 3),
)

# MJCF actuator order: thumb(0-3), index(4-7), middle(8-11), ring(12-15), little(16-19)
# joint_4 of non-thumb fingers is passive (mimic) — the actuator still exists in
# the MJCF but the real SDK only accepts the first 3 joints of each non-thumb finger.
_MJCF_ACTUATOR_NAMES = [
    "right_thumb_joint_1_actuator",  "right_thumb_joint_2_actuator",
    "right_thumb_joint_3_actuator",  "right_thumb_joint_4_actuator",
    "right_index_joint_1_actuator",  "right_index_joint_2_actuator",
    "right_index_joint_3_actuator",  "right_index_joint_4_actuator",
    "right_middle_joint_1_actuator", "right_middle_joint_2_actuator",
    "right_middle_joint_3_actuator", "right_middle_joint_4_actuator",
    "right_ring_joint_1_actuator",   "right_ring_joint_2_actuator",
    "right_ring_joint_3_actuator",   "right_ring_joint_4_actuator",
    "right_little_joint_1_actuator", "right_little_joint_2_actuator",
    "right_little_joint_3_actuator", "right_little_joint_4_actuator",
]


def _mjcf_to_sdk_positions(ctrl: np.ndarray, hand_side: str, pip_scale: float = 1.0) -> List[float]:
    """Convert 20-element MuJoCo ctrl to the 16-element SDK joint list.

    Drops joint_4 of index/middle/ring/little (passive mimic joints).
    pip_scale is applied to joint_3 of the four non-thumb fingers (PIP joints),
    clamped to [0, pi/2] afterwards.
    """
    # ctrl order: thumb(0-3), index(4-7), middle(8-11), ring(12-15), little(16-19)
    # SDK order:  thumb(j1-j4), index(j1-j3), middle(j1-j3), ring(j1-j3), little(j1-j3)
    # PIP = joint_3 of each non-thumb finger: ctrl indices 6, 10, 14, 18
    PIP_UPPER = np.pi / 2  # MJCF ctrlrange upper for joint_3

    def pip(v: float) -> float:
        return float(np.clip(v * pip_scale, 0.0, PIP_UPPER))

    positions: List[float] = []
    # thumb: indices 0-3 (no scaling)
    positions.extend(float(ctrl[i]) for i in range(4))
    # index: j1=4, j2=5, j3=6(pip), skip 7
    positions += [float(ctrl[4]), float(ctrl[5]), pip(ctrl[6])]
    # middle: j1=8, j2=9, j3=10(pip), skip 11
    positions += [float(ctrl[8]), float(ctrl[9]), pip(ctrl[10])]
    # ring: j1=12, j2=13, j3=14(pip), skip 15
    positions += [float(ctrl[12]), float(ctrl[13]), pip(ctrl[14])]
    # little: j1=16, j2=17, j3=18(pip), skip 19
    positions += [float(ctrl[16]), float(ctrl[17]), pip(ctrl[18])]
    return positions


def _build_actuator_index_map(model: mujoco.MjModel, hand_side: str) -> List[int]:
    """Return ctrl indices in MJCF actuator order, adjusted for hand side."""
    prefix = "right" if hand_side == "right" else "left"
    name_to_id = {model.actuator(i).name: i for i in range(model.nu)}
    indices = []
    for name in _MJCF_ACTUATOR_NAMES:
        adjusted = name if hand_side == "right" else name.replace("right_", "left_", 1)
        if adjusted not in name_to_id:
            raise KeyError(f"Actuator '{adjusted}' not found in MuJoCo model.")
        indices.append(name_to_id[adjusted])
    return indices


# ── real hand sender (background thread, ~20 Hz) ─────────────────────────────

class GaiaRealSender:
    def __init__(
        self,
        hand_side: str,
        port: str,
        baudrate: int,
        use_slcan: bool,
        has_main_board: bool,
        speed: float,
        lpf_level: int,
    ):
        try:
            from hand import create_hand
        except ImportError as exc:
            raise ImportError(
                "Gaia HandSDK is required. Install the handsdk wheel and verify "
                "with `python -c \"import hand\"`."
            ) from exc

        self._hand_side = hand_side
        self._speed = speed
        self._lock = threading.Lock()
        self._pending: Optional[List[float]] = None
        self._stop = threading.Event()

        self._hand = create_hand(
            "gaia20",
            hand_side,
            port=port,
            baudrate=baudrate,
            use_slcan=use_slcan,
            has_main_board=has_main_board,
        )
        if not self._hand.connect():
            raise ConnectionError(f"Failed to connect to GaiaHand20 at {port}")
        print(f"[gaia_joint_tuner] Connected to GaiaHand20 ({hand_side}) at {port}", flush=True)

        self._hand.config_pos_lpf_lv(device_id=255, level=int(lpf_level))
        time.sleep(0.5)
        if not self._hand.enable_all_motors_broadcast(True):
            raise RuntimeError("Failed to enable GaiaHand20 motors")
        time.sleep(1.0)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, positions: List[float]) -> None:
        with self._lock:
            self._pending = list(positions)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                positions = self._pending
                self._pending = None
            if positions is not None:
                try:
                    self._hand.move_joints_pos(positions, speed=self._speed, use_broadcast=True)
                except Exception as e:
                    print(f"[gaia_joint_tuner] send error: {e}", flush=True)
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._hand.enable_all_motors_broadcast(False)
        except Exception:
            pass
        try:
            self._hand.close()
        except Exception:
            pass
        print("[gaia_joint_tuner] Disconnected from GaiaHand20.", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GaiaHand20 关节调试工具：MuJoCo 滑条同步真机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hand", type=str, default="right", choices=["right", "left"],
                        help="手的左右侧（默认 right）")
    parser.add_argument("--real", action="store_true",
                        help="同时发送到真机")
    parser.add_argument("--gaia-port", type=str, default="/dev/ttyACM0",
                        help="Gaia 串口（默认 /dev/ttyACM0）")
    parser.add_argument("--gaia-baudrate", type=int, default=921600,
                        help="Gaia 波特率（默认 921600）")
    parser.add_argument("--no-slcan", action="store_true",
                        help="禁用 SLCAN 模式（默认启用 SLCAN）")
    parser.add_argument("--no-main-board", action="store_true",
                        help="声明无主控板（默认有主控板）")
    parser.add_argument("--speed", type=float, default=0.7,
                        help="Gaia 运动速度，范围 (0, 1]（默认 0.7）")
    parser.add_argument("--lpf-level", type=int, default=3, choices=range(0, 6),
                        help="Gaia 电机 LPF 等级 0-5（默认 3）")
    parser.add_argument("--pip-scale", type=float, default=1.0,
                        help="四指 PIP (joint_3) 发给真机的缩放系数，真机弯曲不足时调大（默认 1.0）")
    parser.add_argument("--scene", type=str, default=None,
                        help="手动指定 MuJoCo 场景 XML 路径")
    args = parser.parse_args()

    hand_side = args.hand.lower()

    # resolve scene path
    if args.scene:
        scene_path = Path(args.scene).resolve()
    else:
        candidates = [
            PROJECT_ROOT / "assets" / "gaia_hand20" / f"gaiahand20_{hand_side}_mujoco.xml",
        ]
        scene_path = next((p for p in candidates if p.is_file()), None)
        if scene_path is None:
            print("Could not find MuJoCo scene. Tried:\n" +
                  "\n".join(f"  {p}" for p in candidates))
            sys.exit(1)

    print(f"[gaia_joint_tuner] Scene:    {scene_path}", flush=True)
    print(f"[gaia_joint_tuner] Hand:     {hand_side}", flush=True)
    print(f"[gaia_joint_tuner] PIP scale: {args.pip_scale}", flush=True)

    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)
    actuator_map = _build_actuator_index_map(mj_model, hand_side)

    # real hand
    real_sender: Optional[GaiaRealSender] = None
    if args.real:
        print(f"[gaia_joint_tuner] Connecting to real hand on {args.gaia_port} ...", flush=True)
        real_sender = GaiaRealSender(
            hand_side=hand_side,
            port=args.gaia_port,
            baudrate=args.gaia_baudrate,
            use_slcan=not args.no_slcan,
            has_main_board=not args.no_main_board,
            speed=args.speed,
            lpf_level=args.lpf_level,
        )

    def current_sdk_positions() -> List[float]:
        ctrl = np.array([mj_data.ctrl[i] for i in actuator_map])
        return _mjcf_to_sdk_positions(ctrl, hand_side, args.pip_scale)

    def print_json() -> None:
        ctrl = np.array([mj_data.ctrl[i] for i in actuator_map])
        sdk = _mjcf_to_sdk_positions(ctrl, hand_side, args.pip_scale)
        # SDK order: thumb(j1-j4), index(j1-j3), middle(j1-j3), ring(j1-j3), little(j1-j3)
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        joint_counts = [4, 3, 3, 3, 3]
        record = {}
        idx = 0
        for fname, count in zip(finger_names, joint_counts):
            for j in range(1, count + 1):
                key = f"{hand_side}_{fname}_joint_{j}"
                record[key] = round(sdk[idx], 6)
                idx += 1
        print(json.dumps(record, indent=2), flush=True)

    def key_callback(keycode: int) -> None:
        if keycode in (ord('P'), ord('p')):
            print_json()

    print("[gaia_joint_tuner] Drag sliders in the MuJoCo 'Control' panel.", flush=True)
    print("[gaia_joint_tuner] P = print current joint angles (rad) as JSON", flush=True)

    prev_ctrl = np.array([mj_data.ctrl[i] for i in actuator_map], copy=True)

    try:
        with mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_callback
        ) as viewer:
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[:] = [0.0, 0.0, 0.1]
            viewer.sync()

            while viewer.is_running():
                for _ in range(5):
                    mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                if real_sender is not None:
                    ctrl = np.array([mj_data.ctrl[i] for i in actuator_map])
                    if not np.array_equal(ctrl, prev_ctrl):
                        prev_ctrl = ctrl.copy()
                        real_sender.send(current_sdk_positions())

                time.sleep(0.002)
    finally:
        if real_sender is not None:
            real_sender.close()


if __name__ == "__main__":
    main()
