#!/usr/bin/env python3
"""Display a Luban MuJoCo model and follow its ROS joint states.

The Luban bringup must be running in ``simulation_backend:=mujoco`` first.
Pass the temporary model path printed by bringup with ``--model``.  This
viewer is read-only: it never publishes commands or connects to hardware.
"""

from __future__ import annotations

import argparse
import glob
import threading
import time
from pathlib import Path

import numpy as np


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


def apply_viewer_style(model) -> None:
    """Increase workstation contrast for the dark MuJoCo background."""
    import mujoco
    blue = np.asarray((0.08, 0.32, 0.78, 1.0), dtype=np.float32)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if any(token in name.lower() for token in ("table", "workstation", "pillar")):
            model.geom_rgba[geom_id] = blue
    model.vis.headlight.ambient = np.asarray((0.45, 0.45, 0.45), dtype=np.float32)
    model.vis.headlight.diffuse = np.asarray((0.8, 0.8, 0.8), dtype=np.float32)


def add_blue_floor(viewer) -> None:
    """Add a blue non-colliding reference plane to the viewer scene."""
    import mujoco
    scene = viewer.user_scn
    if scene.maxgeom < 1:
        return
    mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_PLANE,
                        np.asarray((3.0, 3.0, 1.0)), np.asarray((0.0, 0.0, -0.001)),
                        np.eye(3).ravel(), np.asarray((0.08, 0.32, 0.78, 1.0)))
    scene.ngeom = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Luban temporary URDF/MJCF model")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    model_path = resolve_model_path(args.model)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Python is unavailable; use a Python 3.12 ROS environment with mujoco installed"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    apply_viewer_style(model)
    addresses = model_joint_qpos_addresses(model)
    bridge = RosJointStateBridge(model, data, addresses, args.joint_topic)
    print(f"Loaded {model_path} ({model.njnt} joints); following {args.joint_topic}")
    period = 1.0 / args.fps
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            add_blue_floor(viewer)
            while viewer.is_running():
                started = time.perf_counter()
                if not bridge.spin_once():
                    break
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
