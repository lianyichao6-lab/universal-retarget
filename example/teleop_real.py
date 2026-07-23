"""Teleoperation with real robot hand hardware.

Uses the Retargeter interface to map hand tracking input to joint targets and
send them to real hardware via:
- wujihandpy (Wuji Hand)
- TCP socket bridge (Shadow Hand)
- direct serial protocol (Inspire RH56DFX)
- Gaia HandSDK (GaiaHand20 / Gaia20)
"""

import argparse
import json
import pickle
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter


# -------------------- Output drivers --------------------

_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
_INSPIRE_CHANNEL_MAX_RAD = [1.47, 1.47, 1.47, 1.47, 0.6, 1.308]
_INSPIRE_CHANNEL_INVERT = [True, True, True, True, True, True]
_INSPIRE_SERIAL_RESPONSE_TIMEOUT_S = 0.1

_GAIA_FINGER_JOINT_COUNTS = (
    ("thumb", 4),
    ("index", 3),
    ("middle", 3),
    ("ring", 3),
    ("little", 3),
)
_GAIA_MAX_CONSECUTIVE_SEND_FAILURES = 5
class WujiOutput:
    """Output driver for Wuji Hand via wujihandpy."""

    def __init__(self):
        import wujihandpy
        self.hand = wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos, joint_names):
        self.controller.set_joint_target_position(qpos.reshape(5, 4))

    def close(self):
        self.hand.write_joint_enabled(False)


class ShadowTCPOutput:
    """Output driver for Shadow Hand via TCP socket to docker_ros_bridge."""

    def __init__(self, docker_ip="localhost", port=5555):
        self.docker_ip = docker_ip
        self.port = port
        self.sock = self._connect()

    def _connect(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.docker_ip, self.port))
                print(f"Connected to Shadow Hand ROS bridge at {self.docker_ip}:{self.port}")
                return s
            except ConnectionRefusedError:
                print(f"Cannot connect to {self.docker_ip}:{self.port}, retrying in 2s...")
                time.sleep(2)

    def send(self, qpos, joint_names):
        msg = json.dumps({
            "joint_names": joint_names,
            "positions": qpos.tolist(),
        }) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Connection lost, reconnecting...")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = self._connect()
            self.sock.sendall(msg.encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def _inspire_retarget_to_real(retarget_output: np.ndarray) -> list[int]:
    """Map 12-D Inspire retarget output (rad) to 6 serial control channels."""
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.shape[0] < 12:
        raise ValueError(f"Expected at least 12 Inspire joints, got shape {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Inspire retarget output: contains NaN/Inf")

    result: list[int] = []
    for idx, max_rad, invert in zip(
        _INSPIRE_CHANNEL_INDICES,
        _INSPIRE_CHANNEL_MAX_RAD,
        _INSPIRE_CHANNEL_INVERT,
    ):
        value = float(np.clip(output[idx] / max_rad, 0.0, 1.0))
        if invert:
            value = 1.0 - value
        result.append(int(value * 2000))
    return result


def _gaia_retarget_to_real(
    retarget_output: np.ndarray,
    joint_names: list[str],
    hand_side: str,
    pip_scale: float = 1.0,
    thumb_pip_scale: float = 1.0,
) -> list[float]:
    """Map Gaia's 20-joint retarget output to the HandSDK 16-joint order.

    The retargeting URDF contains four revolute joints per finger. For the four
    non-thumb fingers, joint 4 is passive and coupled to joint 3 on the real
    GaiaHand20. HandSDK therefore expects only the active joints in this order:
    thumb(4), index(3), middle(3), ring(3), little(3).
    """
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.ndim != 1:
        raise ValueError(f"Expected 1-D Gaia retarget output, got shape {output.shape}")
    if output.shape[0] != len(joint_names):
        raise ValueError(
            "Gaia qpos/joint-name length mismatch: "
            f"{output.shape[0]} positions vs {len(joint_names)} names"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Gaia retarget output: contains NaN/Inf")
    if not np.isfinite(pip_scale) or pip_scale <= 0.0:
        raise ValueError(f"Gaia PIP scale must be positive, got {pip_scale}")
    if not np.isfinite(thumb_pip_scale) or thumb_pip_scale <= 0.0:
        raise ValueError(
            f"Gaia thumb PIP scale must be positive, got {thumb_pip_scale}"
        )

    name_to_position = dict(zip(joint_names, output))
    if len(name_to_position) != len(joint_names):
        raise ValueError("Gaia retarget joint names contain duplicates")

    sdk_positions: list[float] = []
    missing_names: list[str] = []
    for finger, joint_count in _GAIA_FINGER_JOINT_COUNTS:
        for joint_index in range(1, joint_count + 1):
            name = f"{hand_side}_{finger}_joint_{joint_index}"
            if name not in name_to_position:
                missing_names.append(name)
                continue

            value = float(name_to_position[name])
            if joint_index == 3:
                # Compensate only the real active PIP commands; simulation and
                # retargeting remain unchanged. The four non-thumb DIPs follow
                # their PIPs mechanically, while the thumb DIP remains an
                # independently commanded joint and is not scaled here.
                scale = thumb_pip_scale if finger == "thumb" else pip_scale
                value = float(np.clip(value * scale, 0.0, np.pi / 2.0))
            sdk_positions.append(value)

    if missing_names:
        raise ValueError(
            "Gaia retarget output is missing active joint(s): "
            + ", ".join(missing_names)
        )
    return sdk_positions


class GaiaHand20Output:
    """Output driver for GaiaHand20 through the official Gaia HandSDK."""

    def __init__(
        self,
        port_name: str,
        hand_side: str,
        baudrate: int = 921600,
        use_slcan: bool = True,
        has_main_board: bool = True,
        speed: float = 1.0,
        use_broadcast: bool = True,
        lpf_level: int = 3,
        pip_scale: float = 1.5,
        thumb_pip_scale: float = 1.1,
        enable_delay: float = 1.0,
        command_hz: float = 100.0,
        zero_on_close: bool = False,
    ):
        try:
            from hand import create_hand
        except ImportError as exc:
            raise ImportError(
                "Gaia HandSDK is required for GaiaHand20 control. Install the "
                "handsdk wheel matching your Python/platform from "
                "gaia_hand/02.HandSDK/packages, then verify with "
                "`python -c \"import hand\"`."
            ) from exc

        if not 0.0 < speed <= 1.0:
            raise ValueError(f"Gaia speed must be in (0, 1], got {speed}")
        if not 0 <= lpf_level <= 5:
            raise ValueError(f"Gaia LPF level must be in [0, 5], got {lpf_level}")
        if not np.isfinite(pip_scale) or pip_scale <= 0.0:
            raise ValueError(f"Gaia PIP scale must be positive, got {pip_scale}")
        if not np.isfinite(thumb_pip_scale) or thumb_pip_scale <= 0.0:
            raise ValueError(
                f"Gaia thumb PIP scale must be positive, got {thumb_pip_scale}"
            )
        if command_hz <= 0:
            raise ValueError(f"Gaia command rate must be positive, got {command_hz}")

        self._port_name = port_name
        self._hand_side = hand_side
        self._baudrate = int(baudrate)
        self._use_slcan = bool(use_slcan)
        self._has_main_board = bool(has_main_board)
        self._speed = float(speed)
        self._use_broadcast = bool(use_broadcast)
        self._pip_scale = float(pip_scale)
        self._thumb_pip_scale = float(thumb_pip_scale)
        self._zero_on_close = bool(zero_on_close)
        self._min_command_period = 1.0 / float(command_hz)
        self._last_command_time = -float("inf")
        self._consecutive_send_failures = 0
        self._connected = False
        self._enabled = False

        self.hand = create_hand(
            "gaia20",
            hand_side,
            port=port_name,
            baudrate=self._baudrate,
            use_slcan=self._use_slcan,
            has_main_board=self._has_main_board,
        )

        try:
            if not self.hand.connect():
                raise ConnectionError(
                    f"Failed to connect to GaiaHand20 at {port_name}"
                )
            self._connected = True
            print(
                f"Connected to GaiaHand20 ({hand_side}) at {port_name} "
                f"@ {self._baudrate} baud "
                f"(slcan={self._use_slcan}, main_board={self._has_main_board})."
            )

            self.hand.config_pos_lpf_lv(device_id=255, level=int(lpf_level))
            time.sleep(0.5)

            if not self.hand.enable_all_motors_broadcast(True):
                raise RuntimeError("Failed to enable all GaiaHand20 motors")
            self._enabled = True
            time.sleep(max(0.0, float(enable_delay)))
        except Exception:
            self.close()
            raise

    def send(self, qpos, joint_names):
        now = time.monotonic()
        if now - self._last_command_time < self._min_command_period:
            return

        positions = _gaia_retarget_to_real(
            qpos,
            joint_names,
            self._hand_side,
            pip_scale=self._pip_scale,
            thumb_pip_scale=self._thumb_pip_scale,
        )
        success = self.hand.move_joints_pos(
            positions,
            speed=self._speed,
            use_broadcast=self._use_broadcast,
        )
        self._last_command_time = now

        if success:
            self._consecutive_send_failures = 0
            return

        self._consecutive_send_failures += 1
        if self._consecutive_send_failures >= _GAIA_MAX_CONSECUTIVE_SEND_FAILURES:
            raise RuntimeError(
                "GaiaHand20 rejected "
                f"{self._consecutive_send_failures} consecutive joint commands"
            )
        print(
            "Warning: GaiaHand20 joint command failed "
            f"({self._consecutive_send_failures}/"
            f"{_GAIA_MAX_CONSECUTIVE_SEND_FAILURES})."
        )

    def close(self):
        hand = getattr(self, "hand", None)
        if hand is None:
            return

        try:
            if self._connected and self._zero_on_close:
                print("Returning GaiaHand20 to zero position...")
                hand.hand_zero()
                time.sleep(1.0)
        except Exception as exc:
            print(f"Warning: GaiaHand20 zero-on-close failed: {exc}")

        try:
            if self._connected and self._enabled:
                hand.enable_all_motors_broadcast(False)
        except Exception as exc:
            print(f"Warning: failed to disable GaiaHand20 motors: {exc}")
        finally:
            self._enabled = False

        try:
            hand.close()
        except Exception as exc:
            print(f"Warning: failed to close GaiaHand20 connection: {exc}")
        finally:
            self._connected = False
            self.hand = None
            print(f"Closed GaiaHand20 connection at {self._port_name}.")


class InspireSerialOutput:
    """Direct serial controller for Inspire RH56DFX hand."""

    def __init__(self, port_name: str, baudrate: int = 115200, hand_id: int = 1):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for Inspire hand control. "
                "Install it with `pip install pyserial`."
            ) from exc

        self._port = serial.Serial(port_name, baudrate, timeout=0.01)
        self._hand_id = int(hand_id)
        self._port_name = port_name
        self._baudrate = int(baudrate)
        print(f"Connected to Inspire hand serial at {port_name} @ {baudrate} baud.")

    def _read_response(self) -> bytes:
        deadline = time.time() + _INSPIRE_SERIAL_RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if chunk:
                input_bytes += chunk
            else:
                break
        return bytes(input_bytes)

    @staticmethod
    def _encode_channels(channels: list[int]) -> list[int]:
        if len(channels) != 6:
            raise ValueError(f"Inspire hand expects 6 channels, got {len(channels)}")
        return [int(np.clip(round(ch / 2.0), 0, 1000)) for ch in channels]

    def send(self, qpos, joint_names):
        channels = _inspire_retarget_to_real(qpos)
        encoded = self._encode_channels(channels)
        packet = bytearray([0xEB, 0x90, self._hand_id, 0x0F, 0x12, 0xCE, 0x05])
        for angle in encoded:
            packet.append(angle & 0xFF)
            packet.append((angle >> 8) & 0xFF)
        checksum = sum(packet[2:2 + 0x0F + 3])
        packet.append(checksum & 0xFF)
        self._port.write(packet)
        self._read_response()

    def close(self):
        if self._port.is_open:
            self._port.close()
            print(f"Closed Inspire hand serial at {self._port_name} @ {self._baudrate} baud.")


# -------------------- Teleoperation --------------------

def run_teleop(
    robot_type: str = "wuji",
    hand_side: str = "right",
    config_path: str = "config/mediapipe/mediapipe_wuji_hand.yaml",
    input_device_type: str = "mediapipe_replay",
    visionpro_ip: str = "192.168.50.127",
    quest3_port: int = 9000,
    quest3_protocol: str = "udp",
    pico4_mode: str = "relay",
    pico4_relay_host: str = "127.0.0.1",
    pico4_relay_port: int = 63902,
    pico4_port: int = 63901,
    pico4_broadcast_port: int = 29888,
    noitom_local_ip: str = "192.168.5.25",
    noitom_local_port: int = 8000,
    noitom_server_ip: str = "192.168.5.33",
    noitom_server_port: int = 9000,
    mediapipe_replay_path: str = "data/avp1.pkl",
    video_path: str = "data/right.mp4",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    show_video: bool = False,
    video_depth_scale: float = 1.25,
    docker_ip: str = "localhost",
    docker_port: int = 5555,
    inspire_port: str = "/dev/ttyUSB0",
    inspire_baudrate: int = 115200,
    inspire_hand_id: int = 1,
    gaia_port: str = "/dev/ttyACM0",
    gaia_baudrate: int = 921600,
    gaia_use_slcan: bool = True,
    gaia_has_main_board: bool = True,
    gaia_speed: float = 1.0,
    gaia_use_broadcast: bool = True,
    gaia_lpf_level: int = 3,
    gaia_pip_scale: float = 1.5,
    gaia_thumb_pip_scale: float = 1.1,
    gaia_enable_delay: float = 1.0,
    gaia_command_hz: float = 100.0,
    gaia_zero_on_close: bool = False,
):
    """Run teleoperation with real hardware.

    Input acquisition + retargeting runs in a background thread, while the main
    thread sends the latest joint target to hardware at a steady control rate.
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Import only the selected input backend. Camera/video backends require
    # OpenCV, which should not be mandatory for headset-based teleoperation.
    if input_device_type == "visionpro":
        from input.visionpro import VisionPro
        input_device = VisionPro(ip=visionpro_ip)
    elif input_device_type == "quest3":
        from input.quest3 import Quest3
        input_device = Quest3(port=quest3_port, protocol=quest3_protocol)
    elif input_device_type == "pico4":
        from input.pico4 import Pico4
        input_device = Pico4(
            mode=pico4_mode,
            relay_host=pico4_relay_host,
            relay_port=pico4_relay_port,
            port=pico4_port,
            broadcast_port=pico4_broadcast_port,
        )
    elif input_device_type == "noitom":
        from input.noitom import NoitomInput
        input_device = NoitomInput(
            local_ip=noitom_local_ip,
            local_port=noitom_local_port,
            server_ip=noitom_server_ip,
            server_port=noitom_server_port,
        )
    elif input_device_type == "mediapipe_replay":
        if not mediapipe_replay_path:
            raise ValueError(
                "mediapipe_replay_path is required for mediapipe_replay mode"
            )
        from input.mediapipe_replay import MediaPipeReplay
        input_device = MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        )
    elif input_device_type == "camera":
        from input.camera import Camera
        input_device = Camera(camera_id=0, show_preview=True)
    elif input_device_type == "realsense":
        from input.realsense import Realsense
        input_device = Realsense(hand_side=hand_side, show_video=show_video)
    elif input_device_type == "video":
        from input.video import Video
        input_device = Video(
            video_path=video_path,
            hand_side=hand_side,
            show_video=show_video,
            playback_speed=playback_speed,
            loop=playback_loop,
            depth_scale=video_depth_scale,
        )
    else:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Get joint names from retargeter for name-aware hardware mappings.
    joint_names = retargeter.optimizer.robot.dof_joint_names

    # Initialize hardware only after input/config validation. This avoids
    # leaving an enabled hand behind when an input dependency or YAML is bad.
    if robot_type == "wuji":
        output = WujiOutput()
    elif robot_type == "shadow":
        output = ShadowTCPOutput(docker_ip=docker_ip, port=docker_port)
    elif robot_type == "inspire":
        output = InspireSerialOutput(
            port_name=inspire_port,
            baudrate=inspire_baudrate,
            hand_id=inspire_hand_id,
        )
    elif robot_type == "gaia":
        output = GaiaHand20Output(
            port_name=gaia_port,
            hand_side=hand_side,
            baudrate=gaia_baudrate,
            use_slcan=gaia_use_slcan,
            has_main_board=gaia_has_main_board,
            speed=gaia_speed,
            use_broadcast=gaia_use_broadcast,
            lpf_level=gaia_lpf_level,
            pip_scale=gaia_pip_scale,
            thumb_pip_scale=gaia_thumb_pip_scale,
            enable_delay=gaia_enable_delay,
            command_hz=gaia_command_hz,
            zero_on_close=gaia_zero_on_close,
        )
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    if input_device_type in ("mediapipe_replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    input_data_log = [] if enable_recording else None
    start_time = time.time()

    latest_qpos = np.zeros(retargeter.num_joints, dtype=np.float32)
    qpos_lock = threading.Lock()
    qpos_ready = False
    stop_event = threading.Event()
    input_frame_count = 0
    control_frame_count = 0
    input_thread_error = None

    def input_thread_fn():
        nonlocal qpos_ready, input_frame_count, input_thread_error
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception as exc:
                input_thread_error = exc
                break

            fingers_pose = fingers_data[f"{hand_side}_fingers"]
            if np.allclose(fingers_pose, 0):
                if (
                    input_device_type in ("mediapipe_replay", "video")
                    and not playback_loop
                    and getattr(input_device, "_finished", False)
                ):
                    break
                time.sleep(0.01)
                continue

            if enable_recording and input_data_log is not None:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": fingers_data["left_fingers"].copy(),
                    "right_fingers": fingers_data["right_fingers"].copy(),
                })

            qpos = retargeter.retarget(fingers_pose)
            with qpos_lock:
                latest_qpos[:] = qpos
                qpos_ready = True
            input_frame_count += 1

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print("Starting teleoperation...")
        print(f"  Robot: {robot_type}")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        if input_device_type == "pico4":
            if pico4_mode == "direct":
                print(f"  Pico4 mode: direct (tcp={pico4_port}, udp_broadcast={pico4_broadcast_port})")
            else:
                print(f"  Pico4 mode: relay ({pico4_relay_host}:{pico4_relay_port})")
        if robot_type == "shadow":
            print(f"  Shadow bridge: {docker_ip}:{docker_port}")
        elif robot_type == "inspire":
            print(f"  Inspire serial: {inspire_port} @ {inspire_baudrate} (id={inspire_hand_id})")
        elif robot_type == "gaia":
            print(
                f"  Gaia HandSDK: {gaia_port} @ {gaia_baudrate} "
                f"(slcan={gaia_use_slcan}, main_board={gaia_has_main_board}, "
                f"command_hz={gaia_command_hz:g}, pip_scale={gaia_pip_scale:g}, "
                f"thumb_pip_scale={gaia_thumb_pip_scale:g})"
            )
        print("=" * 50)

        input_thread.start()

        control_hz = 100.0
        control_dt = 1.0 / control_hz
        fps_start_time = time.time()

        while True:
            loop_start = time.time()

            with qpos_lock:
                if qpos_ready:
                    qpos_to_send = latest_qpos.copy()
                else:
                    qpos_to_send = None

            if qpos_to_send is not None:
                output.send(qpos_to_send, joint_names)
                control_frame_count += 1

            if control_frame_count > 0 and control_frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                control_fps = control_frame_count / elapsed
                input_fps = input_frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Control FPS: {control_fps:.1f}  |  Input FPS: {input_fps:.1f}")

            if input_thread_error is not None and not input_thread.is_alive():
                raise RuntimeError("Input/retargeting thread stopped unexpectedly") from input_thread_error

            if (
                input_device_type in ("mediapipe_replay", "video")
                and not playback_loop
                and getattr(input_device, "_finished", False)
                and not input_thread.is_alive()
            ):
                break

            sleep_time = control_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        output.close()

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperation with Real Robot Hand Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wuji Hand with replay data
  python teleop_real.py --robot wuji --play data/avp1.pkl

  # Shadow Hand via TCP to Docker ROS bridge
  python teleop_real.py --robot shadow --play data/avp1.pkl --docker-ip localhost --docker-port 5555

  # Live VisionPro input with Shadow Hand
  python teleop_real.py --robot shadow --input visionpro --ip <your-vision-pro-ip>

  # Inspire Hand via direct serial control
  python teleop_real.py --robot inspire --input noitom --hand right --noitom-local-ip 192.168.5.25 --inspire-port /dev/ttyUSB0

  # Pico 4 direct mode with PC broadcast discovery
  python teleop_real.py --robot inspire --input pico4 --hand right --pico4-mode direct --inspire-port /dev/ttyUSB0

  # GaiaHand20 through HandSDK using the local Pico relay daemon
  python teleop_real.py --robot gaia --input pico4 --hand right --pico4-mode relay --gaia-port /dev/ttyACM0

  # GaiaHand20 without main board (direct serial)
  python teleop_real.py --robot gaia --play data/avp1.pkl --gaia-port /dev/ttyUSB0 --gaia-baudrate 230400 --no-gaia-use-slcan --no-gaia-has-main-board

  # RealSense camera input
  python teleop_real.py --robot shadow --realsense

  # Video file input
  python teleop_real.py --robot shadow --video data/right.mp4

  # Record input data while using VisionPro
  python teleop_real.py --input visionpro --record
        """,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", type=str, default="adaptive",
                        choices=["adaptive", "vector"],
                        help="Optimizer type: adaptive (default) or vector (KeyVectorOptimizer)")
    parser.add_argument("--robot", type=str, default="wuji",
                        choices=["wuji", "shadow", "inspire", "gaia"],
                        help="Robot hand type (default: wuji)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Hand side (default: right)")

    parser.add_argument("--input", type=str, default=None,
                        choices=["visionpro", "quest3", "pico4", "noitom", "mediapipe_replay", "camera", "realsense", "video"],
                        help="Input device type")
    parser.add_argument("--realsense", action="store_true",
                        help="Use RealSense camera (shortcut for --input realsense)")
    parser.add_argument("--show-video", action="store_true",
                        help="Show video with MediaPipe landmarks overlay")
    parser.add_argument("--video-depth-scale", type=float, default=1.25,
                        help="Extra scale applied to MediaPipe z/depth for video input (default: 1.25)")

    parser.add_argument("--play", type=str, default=None, metavar="FILE",
                        help="Play MediaPipe recording file (shortcut for --input mediapipe_replay)")
    parser.add_argument("--video", type=str, default=None, metavar="FILE",
                        help="Play MP4/AVI video file with MediaPipe detection (shortcut for --input video)")

    parser.add_argument("--ip", type=str, default="192.168.50.127",
                        help="VisionPro IP address (default: 192.168.50.127)")
    parser.add_argument("--quest3-port", type=int, default=9000,
                        help="Quest 3 HTS listener port (default: 9000)")
    parser.add_argument("--quest3-protocol", type=str, default="udp", choices=["udp", "tcp"],
                        help="Quest 3 HTS transport protocol (default: udp)")

    parser.add_argument("--pico4-mode", type=str, default="relay", choices=["relay", "direct"],
                        help="Pico 4 input mode: relay daemon (default) or direct TCP server")
    parser.add_argument("--pico4-relay-host", type=str, default="127.0.0.1",
                        help="Pico 4 relay daemon host (default: 127.0.0.1)")
    parser.add_argument("--pico4-relay-port", type=int, default=63902,
                        help="Pico 4 relay daemon port (default: 63902)")
    parser.add_argument("--pico4-port", type=int, default=63901,
                        help="Pico 4 TCP listen port (default: 63901)")
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888,
                        help="Pico 4 direct-mode UDP broadcast port (default: 29888)")

    parser.add_argument("--noitom-local-ip", type=str, default="192.168.5.25",
                        help="Noitom: Linux IP (must match Axis Studio destination, default: 192.168.5.25)")
    parser.add_argument("--noitom-local-port", type=int, default=8000,
                        help="Noitom: local UDP port (default: 7012)")
    parser.add_argument("--noitom-server-ip", type=str, default="192.168.5.33",
                        help="Noitom: Windows Axis Studio IP (default: 192.168.5.33)")
    parser.add_argument("--noitom-server-port", type=int, default=9000,
                        help="Noitom: Axis Studio BVH broadcast port (default: 9000)")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed for replay mode (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable looping for replay/video mode")
    parser.add_argument("--record", action="store_true",
                        help="Record input data to file")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Output file for recording (default: auto-generated)")

    # Shadow Hand TCP options
    parser.add_argument("--docker-ip", type=str, default="localhost",
                        help="Docker ROS bridge IP (default: localhost, for --robot shadow)")
    parser.add_argument("--docker-port", type=int, default=5555,
                        help="Docker ROS bridge port (default: 5555, for --robot shadow)")
    parser.add_argument("--inspire-port", type=str, default="/dev/ttyUSB0",
                        help="Inspire serial port (default: /dev/ttyUSB0, for --robot inspire)")
    parser.add_argument("--inspire-baudrate", type=int, default=115200,
                        help="Inspire serial baudrate (default: 115200, for --robot inspire)")
    parser.add_argument("--inspire-hand-id", type=int, default=1,
                        help="Inspire hand ID in serial protocol (default: 1, for --robot inspire)")

    # GaiaHand20 / HandSDK options
    parser.add_argument("--gaia-port", type=str, default="/dev/ttyACM0",
                        help="GaiaHand20 serial/SLCAN port (default: /dev/ttyACM0)")
    parser.add_argument("--gaia-baudrate", type=int, default=921600,
                        help="GaiaHand20 baudrate (default: 921600; use 230400 without main board)")
    parser.add_argument("--gaia-use-slcan", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Gaia HandSDK SLCAN transport (default: enabled)")
    parser.add_argument("--gaia-has-main-board", action=argparse.BooleanOptionalAction, default=True,
                        help="Declare that the Gaia hand has a main board (default: enabled)")
    parser.add_argument("--gaia-speed", type=float, default=1.0,
                        help="Gaia movement speed in (0, 1] (default/maximum: 1.0)")
    parser.add_argument("--gaia-use-broadcast", action=argparse.BooleanOptionalAction, default=True,
                        help="Use HandSDK broadcast joint commands (default: enabled)")
    parser.add_argument("--gaia-lpf-level", type=int, choices=range(0, 6), default=3,
                        help="Gaia motor position LPF level, 0-5 (default: 3)")
    parser.add_argument("--gaia-pip-scale", type=float, default=1.5,
                        help="Gaia non-thumb PIP command scale (default: 1.5; DIP follows mechanically)")
    parser.add_argument("--gaia-thumb-pip-scale", type=float, default=1.1,
                        help="Gaia thumb PIP (joint_3) command scale (default: 1.1)")
    parser.add_argument("--gaia-enable-delay", type=float, default=1.0,
                        help="Seconds to wait after enabling Gaia motors (default: 1.0)")
    parser.add_argument("--gaia-command-hz", type=float, default=100.0,
                        help="Maximum Gaia command rate in Hz (default/effective maximum: 100)")
    parser.add_argument("--gaia-zero-on-close", action="store_true",
                        help="Return GaiaHand20 to zero before disabling it on exit")


    args = parser.parse_args()

    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")
    if input_device_type == "video" and not video_path:
        parser.error("--video FILE is required for video mode")

    config_path = args.config
    if config_path is None:
        robot_name_map = {
            "wuji": "wuji_hand",
            "shadow": "shadow_hand",
            "inspire": "inspire_hand",
            "gaia": "gaia_hand20",
        }
        input_to_dir = {
            "quest3": "quest3",
            "visionpro": "avp",
            "noitom": "noitom",
            "pico4": "pico4",
        }
        config_dir = input_to_dir.get(input_device_type, "mediapipe")
        robot_file = robot_name_map.get(args.robot, args.robot)
        config_path = f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"

    log = run_teleop(
        robot_type=args.robot,
        hand_side=args.hand,
        config_path=config_path,
        input_device_type=input_device_type,
        visionpro_ip=args.ip,
        quest3_port=args.quest3_port,
        quest3_protocol=args.quest3_protocol,
        pico4_mode=args.pico4_mode,
        pico4_relay_host=args.pico4_relay_host,
        pico4_relay_port=args.pico4_relay_port,
        pico4_port=args.pico4_port,
        pico4_broadcast_port=args.pico4_broadcast_port,
        noitom_local_ip=args.noitom_local_ip,
        noitom_local_port=args.noitom_local_port,
        noitom_server_ip=args.noitom_server_ip,
        noitom_server_port=args.noitom_server_port,
        mediapipe_replay_path=mediapipe_replay_path,
        video_path=video_path,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
        show_video=args.show_video,
        video_depth_scale=args.video_depth_scale,
        docker_ip=args.docker_ip,
        docker_port=args.docker_port,
        inspire_port=args.inspire_port,
        inspire_baudrate=args.inspire_baudrate,
        inspire_hand_id=args.inspire_hand_id,
        gaia_port=args.gaia_port,
        gaia_baudrate=args.gaia_baudrate,
        gaia_use_slcan=args.gaia_use_slcan,
        gaia_has_main_board=args.gaia_has_main_board,
        gaia_speed=args.gaia_speed,
        gaia_use_broadcast=args.gaia_use_broadcast,
        gaia_lpf_level=args.gaia_lpf_level,
        gaia_pip_scale=args.gaia_pip_scale,
        gaia_thumb_pip_scale=args.gaia_thumb_pip_scale,
        gaia_enable_delay=args.gaia_enable_delay,
        gaia_command_hz=args.gaia_command_hz,
        gaia_zero_on_close=args.gaia_zero_on_close,
    )

    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"

        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
