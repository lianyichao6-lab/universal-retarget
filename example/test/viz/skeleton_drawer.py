"""Generic multi-layer skeleton drawing for the AnyDexRetarget tuning viewer."""

from __future__ import annotations

import mujoco
import numpy as np

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_KEYPOINT_RANGES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

DEFAULT_LAYER_CONFIG = {
    "mediapipe_input": {
        "enabled": True,
        "color": [1.0, 0.5, 0.0, 0.6],
        "line_color": [1.0, 0.6, 0.2, 0.5],
        "point_size": 0.004,
        "line_width": 0.002,
    },
    "scaled_target": {
        "enabled": True,
        "color": [0.0, 0.8, 0.8, 0.8],
        "line_color": [0.2, 0.8, 1.0, 0.7],
        "point_size": 0.004,
        "line_width": 0.002,
    },
    "robot_fk": {
        "enabled": True,
        "color": [1.0, 1.0, 1.0, 0.95],
        "line_color": [1.0, 1.0, 1.0, 0.8],
        "point_size": 0.005,
        "line_width": 0.003,
    },
}


class SkeletonDrawer:
    """Draw MediaPipe, scaled-target and generic robot-FK skeleton layers."""

    TIP_INDICES = [4, 8, 12, 16, 20]

    def __init__(self, viz_config: dict | None = None):
        self.highlight_indices: set[int] = set()
        self.update_config(viz_config or {})

    def update_config(self, viz_config: dict) -> None:
        self.layer_configs = {}
        for layer_name in ("mediapipe_input", "scaled_target", "robot_fk"):
            default = DEFAULT_LAYER_CONFIG[layer_name]
            user = viz_config.get(layer_name, {})
            self.layer_configs[layer_name] = {
                "enabled": bool(user.get("enabled", default["enabled"])),
                "color": np.asarray(user.get("color", default["color"]), dtype=np.float32),
                "line_color": np.asarray(
                    user.get("line_color", default["line_color"]), dtype=np.float32
                ),
                "point_size": float(user.get("point_size", default["point_size"])),
                "line_width": float(user.get("line_width", default["line_width"])),
            }
        self.draw_skeleton_lines = bool(viz_config.get("draw_lines", True))
        self.highlight_color = np.asarray(
            viz_config.get("highlight_color", [1.0, 0.0, 0.0, 1.0]),
            dtype=np.float32,
        )

    def set_highlight_fingers(self, finger_names: list[str]) -> None:
        self.highlight_indices = set()
        for name in finger_names:
            if name == "all":
                self.highlight_indices = set(range(21))
                return
            self.highlight_indices.update(FINGER_KEYPOINT_RANGES.get(name, []))

    def clear_highlight(self) -> None:
        self.highlight_indices.clear()

    def draw(
        self,
        scene,
        mediapipe_layer=None,
        scaled_layer=None,
        robot_layer=None,
        pinch_alphas: np.ndarray | None = None,
        axes=None,
    ) -> None:
        """Clear and redraw all enabled layers.

        Each layer is ``(points, connections)`` in MuJoCo world coordinates.
        ``axes`` may be ``(origin, rotation_matrix)``.
        """
        scene.ngeom = 0
        self._draw_layer(scene, "mediapipe_input", mediapipe_layer, highlight=True)
        self._draw_layer(scene, "scaled_target", scaled_layer, highlight=True)
        self._draw_layer(scene, "robot_fk", robot_layer, highlight=False)

        if pinch_alphas is not None and mediapipe_layer is not None:
            self._draw_pinch_indicators(scene, mediapipe_layer[0], pinch_alphas)
        if axes is not None:
            self._draw_axes(scene, axes[0], axes[1])

    def _draw_layer(self, scene, name: str, layer, highlight: bool) -> None:
        cfg = self.layer_configs[name]
        if not cfg["enabled"] or layer is None:
            return
        points, connections = layer
        points = np.asarray(points, dtype=np.float64)
        if self.draw_skeleton_lines:
            self._draw_lines(scene, points, connections, cfg["line_color"], cfg["line_width"])
        self._draw_points(
            scene,
            points,
            cfg["color"],
            cfg["point_size"],
            highlight=highlight,
        )

    def _draw_points(self, scene, positions, color, size, highlight: bool) -> None:
        for index, position in enumerate(positions):
            if scene.ngeom >= scene.maxgeom:
                break
            if not np.all(np.isfinite(position)):
                continue
            draw_color = self.highlight_color if highlight and index in self.highlight_indices else color
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([size, 0.0, 0.0]),
                np.asarray(position, dtype=np.float64),
                np.eye(3).ravel(),
                np.asarray(draw_color, dtype=np.float32),
            )
            scene.ngeom += 1

    @staticmethod
    def _draw_lines(scene, positions, connections, color, width) -> None:
        for start_i, end_i in connections:
            if scene.ngeom >= scene.maxgeom:
                break
            if start_i >= len(positions) or end_i >= len(positions):
                continue
            start = positions[start_i]
            end = positions[end_i]
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                continue
            if np.linalg.norm(end - start) < 1e-7:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([width, 0.0, 0.0]),
                (start + end) / 2.0,
                np.eye(3).ravel(),
                np.asarray(color, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                width,
                start,
                end,
            )
            scene.ngeom += 1

    def _draw_pinch_indicators(self, scene, mediapipe_world, pinch_alphas) -> None:
        alphas = np.asarray(pinch_alphas).reshape(-1)
        for alpha, tip_i in zip(alphas, self.TIP_INDICES):
            if scene.ngeom >= scene.maxgeom or tip_i >= len(mediapipe_world):
                break
            if alpha <= 0.01 or not np.all(np.isfinite(mediapipe_world[tip_i])):
                continue
            color = np.array([1.0, 0.0, 0.0, min(float(alpha), 1.0) * 0.8], dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.006, 0.0, 0.0]),
                np.asarray(mediapipe_world[tip_i], dtype=np.float64),
                np.eye(3).ravel(),
                color,
            )
            scene.ngeom += 1

    def _draw_axes(self, scene, origin, rotation, length=0.04, width=0.0015) -> None:
        colors = (
            np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.9], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float32),
        )
        origin = np.asarray(origin, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        for axis_i, color in enumerate(colors):
            if scene.ngeom >= scene.maxgeom:
                break
            end = origin + rotation[:, axis_i] * length
            self._draw_lines(scene, np.stack([origin, end]), [(0, 1)], color, width)


__all__ = ["SkeletonDrawer", "SKELETON_CONNECTIONS"]
