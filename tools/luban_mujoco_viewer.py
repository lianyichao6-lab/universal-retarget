#!/usr/bin/env python3
"""Display a Luban MuJoCo model or replay a recorded AR5 + L25 episode.

The Luban bringup must be running in ``simulation_backend:=mujoco`` first.
Pass the temporary model path printed by bringup with ``--model``.  This
viewer is read-only: it never publishes commands or connects to hardware.
"""

from __future__ import annotations

import argparse
import glob
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES


def model_joint_qpos_addresses(model) -> dict[str, int]:
    """Return lower-cased MuJoCo joint names to qpos addresses."""
    import mujoco

    result = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result[name.lower()] = int(model.jnt_qposadr[joint_id])
    return result


def apply_joint_state(model, data, addresses: dict[str, int], names, positions) -> int:
    """Apply a ROS JointState sample and return the number of matched joints."""
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("JointState positions must be finite and aligned with names")
    matched = 0
    for name, value in zip(names, values):
        address = addresses.get(str(name).lower())
        if address is None:
            continue
        data.qpos[address] = float(value)
        matched += 1
    return matched

def _load_arm_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a saved Luban-order AR5 trajectory."""
    with np.load(path, allow_pickle=False) as payload:
        names_key = next((key for key in ("joint_names", "arm_joint_names") if key in payload), None)
        positions_key = next((key for key in ("positions", "arm_positions") if key in payload), None)
        if names_key is None or positions_key is None:
            raise ValueError("arm trajectory requires joint_names/positions or arm_joint_names/arm_positions")
        names = tuple(str(name) for name in np.asarray(payload[names_key]).tolist())
        positions = np.asarray(payload[positions_key], dtype=np.float64)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64) if "timestamps" in payload else None
    if names != AR5_RIGHT_JOINT_NAMES:
        raise ValueError(f"AR5 joint names must be {AR5_RIGHT_JOINT_NAMES}, got {names}")
    if positions.ndim != 2 or positions.shape[1] != len(AR5_RIGHT_JOINT_NAMES) or not np.isfinite(positions).all():
        raise ValueError("AR5 positions must be finite with shape (N, 7)")
    if timestamps is not None and (timestamps.shape != (len(positions),) or not np.isfinite(timestamps).all()):
        raise ValueError("AR5 timestamps must be finite with shape (N,)")
    return positions, timestamps


def _load_hand_episode(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load L25 qpos and optional object positions from a replay episode."""
    with np.load(path, allow_pickle=False) as payload:
        if "qpos" not in payload:
            raise ValueError("hand episode requires qpos")
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64) if "timestamps" in payload else None
        object_positions = np.asarray(payload["object_position"], dtype=np.float64) if "object_position" in payload else None
    if qpos.ndim != 2 or qpos.shape[1] != len(L25_QPOS_JOINTS) or not np.isfinite(qpos).all():
        raise ValueError("L25 qpos must be finite with shape (N, 21)")
    if timestamps is not None and (timestamps.shape != (len(qpos),) or not np.isfinite(timestamps).all()):
        raise ValueError("L25 timestamps must be finite with shape (N,)")
    if object_positions is not None and (object_positions.shape != (len(qpos), 3) or not np.isfinite(object_positions).all()):
        raise ValueError("object_position must be finite with shape (N, 3)")
    return qpos, timestamps, object_positions


def load_offline_episode(arm_path: Path, hand_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one synchronized AR5/L25 episode without resampling it."""
    arm, arm_timestamps = _load_arm_trajectory(arm_path)
    hand, hand_timestamps, object_positions = _load_hand_episode(hand_path)
    if len(arm) != len(hand):
        raise ValueError(f"AR5 and L25 frame counts differ: {len(arm)} != {len(hand)}")
    if arm_timestamps is not None and hand_timestamps is not None and not np.allclose(arm_timestamps, hand_timestamps):
        raise ValueError("AR5 and L25 timestamps must match; resample before replay")
    return arm, hand, object_positions


def resolve_model_path(path: Path) -> Path:
    """Resolve an explicit path or the latest live Luban temporary model."""
    if str(path).lower() in {"latest", "auto"}:
        candidates = [Path(item) for item in glob.glob("/tmp/luban_mujoco_robot_*.urdf")]
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime_ns)
        raise FileNotFoundError("no live /tmp/luban_mujoco_robot_*.urdf; start Luban bringup first")
    if path.is_file():
        return path
    raise FileNotFoundError(f"{path} (use exact Luban path, or --model latest while bringup is running)")


class RosJointStateBridge:
    def __init__(self, model, data, addresses, topic: str) -> None:
        try:
            import rclpy
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError(
                "ROS Python dependencies unavailable; source Luban and install mujoco "
                "in the same Python 3.12 environment"
            ) from exc
        self._rclpy = rclpy
        self._model = model
        self._data = data
        self._addresses = addresses
        self._lock = threading.Lock()
        self._matched = 0
        rclpy.init()
        self.node = rclpy.create_node("anydexretarget_luban_mujoco_viewer")
        self.node.create_subscription(JointState, topic, self._callback, 20)

    def _callback(self, message) -> None:
        with self._lock:
            try:
                self._matched = apply_joint_state(
                    self._model, self._data, self._addresses, message.name, message.position
                )
            except ValueError as exc:
                self.node.get_logger().warning(str(exc))

    def spin_once(self) -> bool:
        try:
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            if type(exc).__name__ == "RCLError":
                return False
            raise
        return True

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class OfflineJointStateBridge:
    """No-op bridge for viewing with a MuJoCo-only Python environment."""

    def spin_once(self) -> bool:
        return True

    def close(self) -> None:
        return None


def apply_viewer_style(model) -> None:
    """Keep source materials and improve only neutral scene lighting."""
    model.vis.headlight.ambient = np.asarray((0.45, 0.45, 0.45), dtype=np.float32)
    model.vis.headlight.diffuse = np.asarray((0.8, 0.8, 0.8), dtype=np.float32)


def configure_camera(viewer, model) -> None:
    """Frame the full negative-z Luban workstation in the initial camera."""
    viewer.cam.lookat[:] = np.asarray(model.stat.center, dtype=np.float64)
    viewer.cam.distance = max(float(model.stat.extent) * 2.4, 0.8)


def add_blue_floor(viewer, data) -> None:
    """Add a blue non-colliding reference plane to the viewer scene."""
    import mujoco
    scene = viewer.user_scn
    floor_z = float(np.min(data.geom_xpos[:, 2]) - 0.02)
    if scene.maxgeom < 1:
        return
    mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_PLANE,
                        np.asarray((3.0, 3.0, 1.0)), np.asarray((0.0, 0.0, floor_z)),
                        np.eye(3).ravel(), np.asarray((0.34, 0.44, 0.56, 1.0)))
    scene.ngeom = 1


def prepare_floor_model(model_path: Path, floor_mode: str, floor_z: float) -> Path:
    """Add an optional non-colliding floor to a viewer-only URDF copy."""
    if floor_mode in {"none", "blue"}:
        return model_path
    source = model_path.read_text(encoding="utf-8")
    if "name=\"anydex_checker_floor\"" in source:
        return model_path
    extension = (
        "<mujoco><asset>"
        "<texture name=\"anydex_checker_texture\" type=\"2d\" builtin=\"checker\" "
        "width=\"512\" height=\"512\" rgb1=\"0.12 0.16 0.22\" "
        "rgb2=\"0.68 0.74 0.82\"/>"
        "<material name=\"anydex_checker_material\" texture=\"anydex_checker_texture\" "
        "texrepeat=\"12 12\"/>"
        "</asset><worldbody>"
        f"<geom name=\"anydex_checker_floor\" type=\"plane\" size=\"3 3 0.1\" "
        f"pos=\"0 0 {floor_z:.9g}\" material=\"anydex_checker_material\" "
        "contype=\"0\" conaffinity=\"0\"/>"
        "</worldbody></mujoco>"
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="anydex_luban_checker_", suffix=".urdf", delete=False
    )
    with handle:
        handle.write(source.replace("</robot>", extension + "</robot>"))
    return Path(handle.name)


def prepare_episode_model(
    model_path: Path,
    floor_mode: str,
    floor_z: float,
    object_mesh: Path | None,
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    *,
    object_collision: bool = False,
    floor_collision: bool = False,
) -> Path:
    """Convert Luban URDF to MJCF, then add optional scene geometry.

    MuJoCo's URDF importer accepts the ``<mujoco>`` compiler extension but
    ignores its asset/worldbody additions. The conversion makes those additions
    real MJCF elements before loading the final model.
    """
    import mujoco
    import xml.etree.ElementTree as ET

    handle = tempfile.NamedTemporaryFile(mode="w", prefix="anydex_luban_episode_", suffix=".xml", delete=False)
    handle.close()
    output = Path(handle.name)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mujoco.mj_saveLastXML(str(output), model)
    tree = ET.parse(output)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("balanceinertia", "true")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if floor_mode == "grid":
        ET.SubElement(asset, "texture", {
            "name": "anydex_checker_texture", "type": "2d", "builtin": "checker",
            "width": "512", "height": "512", "rgb1": "0.12 0.16 0.22", "rgb2": "0.68 0.74 0.82",
        })
        ET.SubElement(asset, "material", {
            "name": "anydex_checker_material", "texture": "anydex_checker_texture", "texrepeat": "12 12",
        })
        floor_geometry = {
            "name": "anydex_checker_floor", "type": "plane", "size": "3 3 0.1",
            "pos": f"0 0 {floor_z:.9g}", "material": "anydex_checker_material",
        }
        if not floor_collision:
            floor_geometry.update({"contype": "0", "conaffinity": "0"})
        ET.SubElement(worldbody, "geom", floor_geometry)
    if object_mesh is not None:
        ET.SubElement(asset, "mesh", {"name": "anydex_episode_object_mesh", "file": str(object_mesh.resolve())})
        body = ET.SubElement(worldbody, "body", {
            "name": "anydex_episode_object",
            "pos": " ".join(f"{value:.9g}" for value in object_position),
            "quat": " ".join(f"{value:.9g}" for value in object_quaternion),
        })
        ET.SubElement(body, "freejoint", {"name": "anydex_episode_object_freejoint"})
        geometry = {
            "name": "anydex_episode_object_geom", "type": "mesh", "mesh": "anydex_episode_object_mesh",
            "rgba": "0.12 0.82 0.78 1",
        }
        if not object_collision:
            geometry.update({"contype": "0", "conaffinity": "0"})
        ET.SubElement(body, "geom", geometry)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def apply_offline_frame(model, data, addresses: dict[str, int], arm: np.ndarray, hand: np.ndarray, object_positions: np.ndarray | None, index: int) -> None:
    """Apply one saved AR5+L25 frame to the full Luban model."""
    apply_joint_state(model, data, addresses, AR5_RIGHT_JOINT_NAMES, arm[index])
    apply_joint_state(model, data, addresses, tuple(f"r_hand_{name}" for name in L25_QPOS_JOINTS), hand[index])
    if object_positions is not None:
        object_address = addresses.get("anydex_episode_object_freejoint")
        if object_address is None:
            raise ValueError("object trajectory was provided but --object-mesh was not set")
        data.qpos[object_address:object_address + 3] = object_positions[index]


def configure_render_camera(camera, model) -> None:
    """Use full-workstation framing for headless rendering."""
    import mujoco

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(model.stat.center, dtype=np.float64)
    camera.distance = max(float(model.stat.extent) * 2.4, 0.8)
    camera.azimuth = 145.0
    camera.elevation = -22.0


def render_offline_video(model, data, addresses, arm, hand, object_positions, output: Path, fps: float, width: int, height: int) -> None:
    """Render one full-robot trajectory through ffmpeg without a GUI."""
    import subprocess
    import mujoco

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:.9g}", "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            camera = mujoco.MjvCamera()
            configure_render_camera(camera, model)
            for index in range(len(arm)):
                apply_offline_frame(model, data, addresses, arm, hand, object_positions, index)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                assert process.stdin is not None
                process.stdin.write(np.ascontiguousarray(renderer.render(), dtype=np.uint8).tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Luban temporary URDF/MJCF model")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--floor", choices=("grid", "blue", "none"), default="grid", help="Viewer-only floor style (default: grid).")
    parser.add_argument("--arm-trajectory", type=Path, help="Offline AR5 trajectory NPZ in Luban joint order.")
    parser.add_argument("--hand-episode", type=Path, help="Offline L25 replay NPZ containing qpos (N,21).")
    parser.add_argument("--object-mesh", type=Path, help="Object mesh already expressed in AR5 base coordinates.")
    parser.add_argument("--object-position", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--object-quaternion", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--record", type=Path, help="Write a full-robot MP4 from an offline episode.")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--no-viewer", action="store_true", help="Render only; do not open the MuJoCo window.")
    parser.add_argument("--loop", action="store_true", help="Loop an offline episode in the interactive viewer.")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.render_width <= 0 or args.render_height <= 0:
        parser.error("render dimensions must be positive")
    offline_requested = args.arm_trajectory is not None or args.hand_episode is not None
    if (args.arm_trajectory is None) != (args.hand_episode is None):
        parser.error("--arm-trajectory and --hand-episode must be supplied together")
    if args.record is not None and not offline_requested:
        parser.error("--record requires --arm-trajectory and --hand-episode")
    if args.no_viewer and not offline_requested:
        parser.error("--no-viewer requires an offline episode")
    if args.object_mesh is not None and not args.object_mesh.is_file():
        parser.error(f"--object-mesh does not exist: {args.object_mesh}")
    if offline_requested and args.object_mesh is None:
        parser.error("offline AR5+L25 replay requires --object-mesh in AR5 base coordinates")
    model_path = resolve_model_path(args.model)
    arm = hand = object_positions = None
    if offline_requested:
        arm, hand, object_positions = load_offline_episode(args.arm_trajectory, args.hand_episode)
        if object_positions is None:
            object_positions = np.repeat(np.asarray(args.object_position, dtype=np.float64)[None], len(arm), axis=0)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Python is unavailable; use a Python 3.12 ROS environment with mujoco installed"
        ) from exc

    probe_model = mujoco.MjModel.from_xml_path(str(model_path))
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    floor_z = float(np.min(probe_data.geom_xpos[:, 2]) - 0.02)
    floor_source = prepare_episode_model(
        model_path, args.floor, floor_z, args.object_mesh,
        np.asarray(args.object_position, dtype=np.float64),
        np.asarray(args.object_quaternion, dtype=np.float64),
    )
    model = mujoco.MjModel.from_xml_path(str(floor_source))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    apply_viewer_style(model)
    addresses = model_joint_qpos_addresses(model)
    if offline_requested:
        bridge = OfflineJointStateBridge()
        bridge_mode = f"offline AR5+L25 replay ({len(arm)} frames)"
    else:
        try:
            bridge = RosJointStateBridge(model, data, addresses, args.joint_topic)
            bridge_mode = f"following {args.joint_topic}"
        except RuntimeError as exc:
            print(f"ROS bridge unavailable; offline viewer mode: {exc}")
            bridge = OfflineJointStateBridge()
            bridge_mode = "offline"
    print(f"Loaded {model_path} ({model.njnt} joints); {bridge_mode}")
    if args.record is not None:
        render_offline_video(model, data, addresses, arm, hand, object_positions, args.record, args.fps, args.render_width, args.render_height)
        print(f"Wrote full-robot replay: {args.record}")
    if args.no_viewer:
        bridge.close()
        return 0
    period = 1.0 / args.fps
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            configure_camera(viewer, model)
            if args.floor == "blue":
                add_blue_floor(viewer, data)
            index = 0
            while viewer.is_running():
                started = time.perf_counter()
                if offline_requested:
                    apply_offline_frame(model, data, addresses, arm, hand, object_positions, index)
                    index += 1
                    if index == len(arm):
                        index = 0 if args.loop else len(arm) - 1
                elif not bridge.spin_once():
                    break
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
