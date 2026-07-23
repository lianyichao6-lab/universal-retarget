"""Interactive MuJoCo viewer for AnyDexRetarget parameter tuning.

This module is adapted from the reference Wuji tuning viewer, but uses the
current project's generic robot configuration, qpos mapping and optimizer link
metadata so it works with all hands listed in ``example/teleop_sim.py``.
"""

from __future__ import annotations

import pickle
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter

# Use the same model metadata and qpos/actuator mapping as teleop_sim.py and
# debug_skeleton.py. Importing this dedicated module does not pull in camera or
# OpenCV input dependencies.
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    validate_mujoco_actuator_mapping,
)

from .config_watcher import ConfigWatcher, _diff_configs
from .param_map import get_affected_fingers
from .skeleton_drawer import SKELETON_CONNECTIONS, SkeletonDrawer

_MP_FINGER_JOINTS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


def _compute_scaled_mediapipe(mediapipe_kp: np.ndarray, optimizer) -> np.ndarray:
    """Build the target skeleton represented by the current scale parameters."""
    mediapipe_kp = np.asarray(mediapipe_kp, dtype=np.float64)
    wrist = mediapipe_kp[0]
    scaled = mediapipe_kp.copy()

    scaling_matrix = getattr(optimizer, "segment_scaling_full", None)
    if scaling_matrix is not None:
        scaling_matrix = np.asarray(scaling_matrix, dtype=np.float64)
        mp_fingers = list(getattr(optimizer, "mp_finger_indices", range(len(scaling_matrix))))
        for local_i, mp_finger_i in enumerate(mp_fingers):
            if local_i >= len(scaling_matrix) or mp_finger_i >= len(_MP_FINGER_JOINTS):
                continue
            for point_i, mp_joint_i in enumerate(_MP_FINGER_JOINTS[mp_finger_i]):
                column = min(point_i, scaling_matrix.shape[1] - 1)
                scale = scaling_matrix[local_i, column]
                scaled[mp_joint_i] = wrist + (mediapipe_kp[mp_joint_i] - wrist) * scale
        return scaled

    global_scale = float(getattr(optimizer, "scaling", 1.0))
    scaled[1:] = wrist + (mediapipe_kp[1:] - wrist) * global_scale
    return scaled


def _robot_fk_skeleton(optimizer, qpos: np.ndarray):
    """Return generic robot FK points and chain connections in Pinocchio space."""
    robot = optimizer.robot
    robot.compute_forward_kinematics(np.asarray(qpos, dtype=np.float64))

    def point(link_name: str, offset=None) -> np.ndarray:
        link_id = robot.get_link_index(link_name)
        pose = robot.get_link_pose(link_id)
        position = pose[:3, 3].copy()
        if offset is not None:
            position += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
        return position

    points = [point(optimizer.origin_link_name)]
    connections: list[tuple[int, int]] = []
    for local_i in range(optimizer.num_fingers):
        chain = []
        if local_i < len(getattr(optimizer, "link1_names", [])):
            chain.append((optimizer.link1_names[local_i], None))
        if local_i < len(getattr(optimizer, "link3_names", [])):
            offsets = getattr(optimizer, "link3_offsets", None)
            chain.append((optimizer.link3_names[local_i], offsets[local_i] if offsets is not None else None))
        if local_i < len(getattr(optimizer, "link4_names", [])):
            offsets = getattr(optimizer, "link4_offsets", None)
            chain.append((optimizer.link4_names[local_i], offsets[local_i] if offsets is not None else None))
        if local_i < len(optimizer.task_link_names):
            offsets = getattr(optimizer, "task_offsets", None)
            chain.append((optimizer.task_link_names[local_i], offsets[local_i] if offsets is not None else None))

        previous = 0
        for link_name, offset in chain:
            current = len(points)
            points.append(point(link_name, offset))
            connections.append((previous, current))
            previous = current
    return np.asarray(points), connections


class TuningViewer:
    """Three-layer tuning viewer with YAML hot reload."""

    def __init__(
        self,
        hand_side: str = "left",
        retarget_config_path: str | None = None,
        viz_config_path: str | None = None,
        mjcf_path: str | None = None,
    ):
        self.hand_side = hand_side.lower()
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be left/right, got {hand_side}")

        if retarget_config_path is None:
            retarget_config_path = str(
                EXAMPLE_ROOT / "config/adaptive/avp/avp_wuji_hand.yaml"
            )
        self.retarget_config_path = Path(retarget_config_path).expanduser().resolve()
        if not self.retarget_config_path.exists():
            raise FileNotFoundError(f"Retarget config not found: {self.retarget_config_path}")

        self.retarget_config = self._load_yaml(self.retarget_config_path)
        self.retargeter = Retargeter.from_config(
            self.retarget_config.copy(), hand_side=self.hand_side
        )
        self.robot_type = self.retarget_config.get("robot", {}).get("type", "shadow_hand")
        if self.robot_type not in ROBOT_HAND_CONFIGS:
            raise ValueError(
                f"No MuJoCo viewer configuration for {self.robot_type!r}; "
                f"supported: {sorted(ROBOT_HAND_CONFIGS)}"
            )
        self.hand_cfg = ROBOT_HAND_CONFIGS[self.robot_type]

        self.viz_config = {}
        if viz_config_path:
            viz_path = Path(viz_config_path).expanduser().resolve()
            if not viz_path.exists():
                raise FileNotFoundError(f"Visualization config not found: {viz_path}")
            self.viz_config = self._load_yaml(viz_path)

        model_path = Path(mjcf_path).expanduser().resolve() if mjcf_path else Path(
            self.hand_cfg["model_path"](self.hand_side)
        ).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        validate_mujoco_actuator_mapping(self.model, self.hand_cfg)

        mesh_alpha = float(self.viz_config.get("robot_mesh", {}).get("alpha", 0.3))
        hand_geom_mask = self.model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        self.model.geom_rgba[hand_geom_mask, 3] = mesh_alpha

        self.drawer = SkeletonDrawer(self.viz_config.get("skeleton", {}))
        self.config_watcher = ConfigWatcher(
            str(self.retarget_config_path),
            poll_interval=float(
                self.viz_config.get("hot_reload", {}).get("poll_interval", 0.5)
            ),
            verbose=True,
        )
        self._highlight_timer = 0.0
        self._highlight_duration = float(
            self.viz_config.get("skeleton", {}).get("highlight_duration", 1.0)
        )
        self._mujoco_origin_body_id = self._find_mujoco_origin_body()
        self._initialize_model()

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _find_mujoco_origin_body(self) -> int:
        optimizer = self.retargeter.optimizer
        candidates = [
            optimizer.origin_link_name,
            f"{self.hand_side}_{optimizer.origin_link_name}",
            "palm_link",
            "base_link",
            "base",
            "hand",
        ]
        for name in candidates:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id

        roots = [
            body_id for body_id in range(1, self.model.nbody)
            if self.model.body_parentid[body_id] == 0
        ]
        if len(roots) == 1:
            return roots[0]
        # Some hand-only MJCFs attach the palm geoms and every finger root
        # directly to world (for example assets/wuji_hand/right.xml).  In that
        # layout the optimizer palm frame corresponds to MuJoCo world.
        if len(roots) > 1:
            return 0
        raise RuntimeError("MuJoCo model has no usable robot origin body")

    def _initialize_model(self) -> None:
        if self.model.nu > 0:
            for actuator_i in range(self.model.nu):
                if self.model.actuator_ctrllimited[actuator_i]:
                    lower, upper = self.model.actuator_ctrlrange[actuator_i]
                    self.data.ctrl[actuator_i] = (lower + upper) / 2.0
                else:
                    self.data.ctrl[actuator_i] = 0.0
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)

    def _set_camera(self, viewer) -> None:
        camera = self.viz_config.get("camera", {})
        fallback = self.retarget_config.get("render", {}).get("camera", {})
        viewer.cam.azimuth = camera.get("azimuth", fallback.get("azimuth", 135))
        viewer.cam.elevation = camera.get("elevation", fallback.get("elevation", -20))
        viewer.cam.distance = camera.get("distance", fallback.get("distance", 0.5))
        viewer.cam.lookat[:] = camera.get("lookat", fallback.get("lookat", [0.0, 0.0, 0.05]))

    def _reload_retargeter(self, new_config: dict, changes: list) -> None:
        new_robot_type = new_config.get("robot", {}).get("type", "shadow_hand")
        if new_robot_type != self.robot_type:
            print(
                "[TuningViewer] robot.type changed. Restart the viewer to load "
                "the new MuJoCo model."
            )
            return
        self.retarget_config = new_config
        self.retargeter = Retargeter.from_config(new_config.copy(), self.hand_side)
        self.retargeter.reset_filter()

        affected = set()
        for parameter_path, _, _ in changes:
            lookup = parameter_path.removeprefix("retarget.")
            affected.update(get_affected_fingers(lookup))
        if affected:
            self.drawer.set_highlight_fingers(sorted(affected))
            self._highlight_timer = time.time()

    def _check_highlight_timeout(self) -> None:
        if self._highlight_timer and time.time() - self._highlight_timer > self._highlight_duration:
            self.drawer.clear_highlight()
            self._highlight_timer = 0.0

    def _process_frame(self, raw_keypoints: np.ndarray) -> dict:
        qpos, verbose = self.retargeter.retarget_verbose(raw_keypoints)
        transformed = verbose["mediapipe_kp"]
        return {
            "qpos": qpos,
            "mediapipe_kp": transformed,
            "scaled_kp": _compute_scaled_mediapipe(transformed, self.retargeter.optimizer),
            "cost": verbose.get("cost", 0.0),
            "pinch_alphas": verbose.get("pinch_alphas"),
        }

    def apply_result(self, result: dict) -> None:
        """Apply one retarget result to the MuJoCo model."""
        target = map_retarget_qpos(
            result["qpos"],
            self.hand_cfg,
            self.retargeter.optimizer.robot.dof_joint_names,
        )
        target = np.asarray(target, dtype=np.float64)
        direct = bool(self.hand_cfg.get("direct_qpos", False))
        qpos_servo = self.hand_cfg.get("qpos_servo_alpha") is not None

        if direct or qpos_servo or self.model.nu == 0:
            count = min(len(target), self.model.nq)
            self.data.qpos[:count] = target[:count]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            count = min(len(target), self.model.nu)
            self.data.ctrl[:count] = target[:count]
            for _ in range(8):
                mujoco.mj_step(self.model, self.data)

    def _alignment(self):
        """Return Pinocchio->MuJoCo rigid alignment for the robot origin."""
        optimizer = self.retargeter.optimizer
        robot = optimizer.robot
        origin_id = robot.get_link_index(optimizer.origin_link_name)
        pin_pose = robot.get_link_pose(origin_id)
        pin_position = pin_pose[:3, 3].copy()
        pin_rotation = pin_pose[:3, :3].copy()

        mujoco.mj_forward(self.model, self.data)
        mj_position = self.data.xpos[self._mujoco_origin_body_id].copy()
        mj_rotation = self.data.xmat[self._mujoco_origin_body_id].reshape(3, 3).copy()
        frame_rotation = mj_rotation @ pin_rotation.T
        return pin_position, mj_position, frame_rotation

    @staticmethod
    def _place_input_skeleton(keypoints: np.ndarray, pin_origin: np.ndarray) -> np.ndarray:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        return pin_origin + (keypoints - keypoints[0])

    @staticmethod
    def _align_points(points, pin_origin, mj_origin, frame_rotation):
        return (np.asarray(points) - pin_origin) @ frame_rotation.T + mj_origin

    def draw_result(self, scene, result: dict) -> None:
        """Draw all three skeleton layers for one processed frame."""
        fk_points, fk_connections = _robot_fk_skeleton(
            self.retargeter.optimizer, result["qpos"]
        )
        pin_origin, mj_origin, frame_rotation = self._alignment()

        input_pin = self._place_input_skeleton(result["mediapipe_kp"], pin_origin)
        scaled_pin = self._place_input_skeleton(result["scaled_kp"], pin_origin)
        input_world = self._align_points(input_pin, pin_origin, mj_origin, frame_rotation)
        scaled_world = self._align_points(scaled_pin, pin_origin, mj_origin, frame_rotation)
        fk_world = self._align_points(fk_points, pin_origin, mj_origin, frame_rotation)

        self.drawer.draw(
            scene,
            mediapipe_layer=(input_world, SKELETON_CONNECTIONS),
            scaled_layer=(scaled_world, SKELETON_CONNECTIONS),
            robot_layer=(fk_world, fk_connections),
            pinch_alphas=result.get("pinch_alphas"),
            axes=(mj_origin, frame_rotation),
        )

    def check_config_reload(self) -> bool:
        changed, new_config = self.config_watcher.check()
        if changed:
            changes = _diff_configs(self.retarget_config, new_config)
            self._reload_retargeter(new_config, changes)
            self.retargeter.reset()
        self._check_highlight_timeout()
        return changed

    def play_recording(
        self,
        data_or_path,
        fps: float = 30.0,
        hand_key: str | None = None,
        trust_pkl: bool = False,
    ) -> None:
        if isinstance(data_or_path, (str, Path)):
            if not trust_pkl:
                raise ValueError(
                    "Refusing to load pickle without explicit trust. "
                    "Set trust_pkl=True only for trusted data."
                )
            with Path(data_or_path).open("rb") as file:
                data = pickle.load(file)
        else:
            data = data_or_path
        if not data:
            raise ValueError("No recording frames to play")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        hand_key = hand_key or f"{self.hand_side}_fingers"

        total_frames = len(data)
        current_frame = 0
        last_result = None
        running = True

        def stop(_signal, _frame):
            nonlocal running
            running = False

        old_handler = signal.signal(signal.SIGINT, stop)
        print("=" * 60)
        print("AnyDexRetarget Tuning Viewer")
        print(f"  Robot: {self.robot_type}")
        print(f"  Model: {self.model_path}")
        print(f"  Config: {self.retarget_config_path}")
        print("  Orange=input, Cyan=scaled target, White=robot FK")
        print("  Edit the YAML file while running to hot-reload parameters")
        print("=" * 60)

        frame_period = 1.0 / fps
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._set_camera(viewer)
                while viewer.is_running() and running:
                    start = time.perf_counter()
                    changed = self.check_config_reload()
                    frame = data[current_frame]
                    raw = frame.get(hand_key) if isinstance(frame, dict) else None
                    if raw is not None and np.asarray(raw).shape == (21, 3) and not np.allclose(raw, 0):
                        last_result = self._process_frame(np.asarray(raw))
                        self.apply_result(last_result)
                    current_frame = (current_frame + 1) % total_frames

                    if changed and last_result is not None:
                        # Current frame is already processed above with the new retargeter.
                        pass
                    if last_result is not None:
                        with viewer.lock():
                            self.draw_result(viewer.user_scn, last_result)
                    viewer.sync()
                    elapsed = time.perf_counter() - start
                    if elapsed < frame_period:
                        time.sleep(frame_period - elapsed)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        print(f"Viewer closed at frame {current_frame}")

    def view_single_frame(self, raw_keypoints: np.ndarray) -> None:
        result = self._process_frame(raw_keypoints)
        self.apply_result(result)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self._set_camera(viewer)
            while viewer.is_running():
                if self.check_config_reload():
                    result = self._process_frame(raw_keypoints)
                    self.apply_result(result)
                with viewer.lock():
                    self.draw_result(viewer.user_scn, result)
                viewer.sync()
                time.sleep(0.03)


__all__ = ["TuningViewer"]
