"""Unified retargeting interface for Wuji Hand.

Provides a high-level interface that handles:
- MediaPipe coordinate transformation
- Optional rotation adjustment
- Optimizer selection (TipDirVec/FullHandVec/Adaptive)
- Low-pass filtering

Usage:
    retargeter = Retargeter.from_yaml("config/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from .optimizer import BaseOptimizer, LPFilter
from .mediapipe import apply_mediapipe_transformations


def _resolve_hand_side_config(value, hand_side: str):
    """Resolve an optional left/right configuration section.

    Flat dictionaries remain backward compatible.  A side-aware dictionary may
    contain shared keys plus ``default``, ``left`` and ``right`` sections; the
    selected hand section overrides the shared/default values.
    """
    if not isinstance(value, dict) or not ({"left", "right", "default"} & set(value)):
        return value

    shared = {
        key: item
        for key, item in value.items()
        if key not in {"left", "right", "default"}
    }
    default = value.get("default", {})
    side_value = value.get(hand_side, {})
    if not isinstance(default, dict):
        raise ValueError("Side-aware config field 'default' must be a dictionary")
    if not isinstance(side_value, dict):
        raise ValueError(
            f"Side-aware config field '{hand_side}' must be a dictionary"
        )
    return {**shared, **default, **side_value}


class Retargeter:
    """Unified retargeting interface for Wuji Hand.

    Encapsulates the complete retargeting pipeline:
    1. MediaPipe coordinate transformation (raw -> wrist frame)
    2. Optional rotation adjustment (configured via mediapipe_rotation)
    3. IK optimization (TipDirVec/FullHandVec/Adaptive)
    4. Low-pass filtering for smooth output

    Attributes:
        optimizer: The underlying optimizer instance
        lp_filter: Low-pass filter for smoothing
        hand_side: 'left' or 'right'
    """

    def __init__(self, config: dict, hand_side: str = "right"):
        """Initialize retargeter.

        Args:
            config: Configuration dict (from YAML)
            hand_side: 'left' or 'right'
        """
        self.config = config
        self.hand_side = hand_side.lower()

        if self.hand_side not in ['left', 'right']:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")

        # Ensure hand_side in config
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['hand_side'] = self.hand_side

        # Create optimizer
        self.optimizer = BaseOptimizer.from_config(config)

        # Create low-pass filter
        retarget_config = config.get('retarget', {})
        lp_alpha = retarget_config.get('lp_alpha', 0.2)
        self.lp_filter = LPFilter(lp_alpha)

        # Robot-specific preprocessing switches.
        self.robot_type = config.get('robot', {}).get('type', 'shadow_hand')

        # Rotation and calibrated landmark-group offsets.  Offsets are
        # configured in centimetres to match calibrate_offset.py output, then
        # converted once here because all optimizer coordinates use metres.
        self.rotation_xyz = _resolve_hand_side_config(
            retarget_config.get('mediapipe_rotation', {}), self.hand_side
        )
        self.wrist_offset_m = self._parse_offset_cm(
            retarget_config.get('wrist_offset_cm', [0.0, 0.0, 0.0]),
            'wrist_offset_cm',
        )
        self.thumb_offset_m = self._parse_offset_cm(
            retarget_config.get('thumb_offset_cm', [0.0, 0.0, 0.0]),
            'thumb_offset_cm',
        )
        self.align_four_mcp_to_robot = bool(
            retarget_config.get('align_four_mcp_to_robot', False)
        )
        self._robot_mcp_targets = self._compute_robot_mcp_targets()

    @classmethod
    def from_yaml(cls, yaml_path: str, hand_side: str = "right") -> "Retargeter":
        """Create retargeter from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file
            hand_side: 'left' or 'right'

        Returns:
            Retargeter instance
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config, hand_side)

    @classmethod
    def from_config(cls, config: dict, hand_side: str = "right") -> "Retargeter":
        """Create retargeter from configuration dict.

        Args:
            config: Configuration dict
            hand_side: 'left' or 'right'

        Returns:
            Retargeter instance
        """
        return cls(config, hand_side)

    @staticmethod
    def _parse_offset_cm(value, name: str) -> np.ndarray:
        """Validate a three-axis centimetre offset and convert it to metres."""
        offset = np.asarray(value, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError(f"{name} must contain exactly 3 values, got {offset.shape}")
        if not np.all(np.isfinite(offset)):
            raise ValueError(f"{name} must contain only finite values, got {value}")
        return offset / 100.0

    def _compute_robot_mcp_targets(self) -> dict[int, np.ndarray]:
        """Cache fixed robot MCP mount positions in the optimizer frame."""
        if not self.align_four_mcp_to_robot:
            return {}
        if self.robot_type != 'gaia_hand20':
            return {}

        optimizer = self.optimizer
        robot = optimizer.robot
        qpos = optimizer.neutral_qpos.copy() if optimizer.neutral_qpos is not None \
            else np.zeros(robot.model.nq, dtype=np.float64)
        robot.compute_forward_kinematics(qpos)
        origin_id = robot.get_link_index(optimizer.origin_link_name)
        origin = robot.get_link_pose(origin_id)[:3, 3].copy()

        targets = {}
        mcp_indices = [1, 5, 9, 13, 17]
        for local_i, finger_i in enumerate(optimizer.mp_finger_indices):
            if finger_i == 0 or local_i >= len(optimizer.link1_names):
                continue
            link_id = robot.get_link_index(optimizer.link1_names[local_i])
            link_pos = robot.get_link_pose(link_id)[:3, 3].copy()
            targets[mcp_indices[finger_i]] = link_pos - origin
        return targets

    def preprocess_transformed_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply robot-specific preprocessing to wrist-frame keypoints.

        The returned array is the exact input passed to ``optimizer.solve``.
        """
        processed = np.asarray(keypoints, dtype=np.float64).copy()
        # Match the reference calibrator: one common correction for all four
        # non-thumb chains and one correction for the thumb chain.
        processed[5:] += self.wrist_offset_m
        processed[1:5] += self.thumb_offset_m

        # Explicit robot-MCP alignment, when enabled, intentionally wins over
        # the group offset for the MCP landmarks it replaces.
        if self._robot_mcp_targets:
            wrist = processed[0].copy()
            for kp_index, target in self._robot_mcp_targets.items():
                processed[kp_index] = wrist + target
        return processed

    def retarget(
        self,
        raw_keypoints: np.ndarray,
        apply_filter: bool = True,
    ) -> np.ndarray:
        """Retarget raw MediaPipe keypoints to joint angles.

        Args:
            raw_keypoints: (21, 3) raw MediaPipe keypoints
            apply_filter: Whether to apply low-pass filter

        Returns:
            qpos: (20,) joint angles
        """
        # Apply coordinate transformation
        mediapipe_kp = apply_mediapipe_transformations(raw_keypoints, self.hand_side)

        # Apply rotation adjustment if configured
        if self.rotation_xyz:
            mediapipe_kp = self._apply_rotation(mediapipe_kp)

        mediapipe_kp = self.preprocess_transformed_keypoints(mediapipe_kp)

        # Solve IK
        qpos = self.optimizer.solve(mediapipe_kp)

        # Apply filter
        if apply_filter:
            qpos = self.lp_filter.next(qpos)

        return qpos

    def retarget_verbose(
        self,
        raw_keypoints: np.ndarray,
        apply_filter: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Retarget with verbose output for visualization.

        Args:
            raw_keypoints: (21, 3) raw MediaPipe keypoints
            apply_filter: Whether to apply low-pass filter

        Returns:
            Tuple of (qpos, verbose_dict) where verbose_dict contains:
                - mediapipe_kp: Transformed keypoints
                - qpos_unfiltered: Joint angles before filtering
                - qpos: Final joint angles
                - cost: Optimization cost
                - pinch_alphas: (AdaptiveOptimizer only) alpha values
        """
        # Apply coordinate transformation
        mediapipe_kp = apply_mediapipe_transformations(raw_keypoints, self.hand_side)

        # Apply rotation adjustment if configured
        if self.rotation_xyz:
            mediapipe_kp = self._apply_rotation(mediapipe_kp)

        mediapipe_kp = self.preprocess_transformed_keypoints(mediapipe_kp)

        # Solve IK
        qpos = self.optimizer.solve(mediapipe_kp)

        # Apply filter
        if apply_filter:
            filtered_qpos = self.lp_filter.next(qpos)
        else:
            filtered_qpos = qpos

        # Build verbose dict
        verbose_dict = {
            'mediapipe_kp': mediapipe_kp.copy(),
            'qpos_unfiltered': qpos.copy(),
            'qpos': filtered_qpos.copy(),
            'cost': self.optimizer.compute_cost(qpos, mediapipe_kp),
        }

        # Add optimizer-specific data
        if hasattr(self.optimizer, '_compute_pinch_alpha'):
            verbose_dict['pinch_alphas'] = self.optimizer._compute_pinch_alpha(mediapipe_kp)

        return filtered_qpos, verbose_dict

    def _apply_rotation(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply rotation adjustment to keypoints.

        Uses extrinsic XYZ Euler angles (rotations around fixed axes).

        Args:
            keypoints: (21, 3) keypoints in wrist frame

        Returns:
            Rotated keypoints
        """
        from scipy.spatial.transform import Rotation

        x_deg = self.rotation_xyz.get('x', 0.0)
        y_deg = self.rotation_xyz.get('y', 0.0)
        z_deg = self.rotation_xyz.get('z', 0.0)

        if x_deg == 0 and y_deg == 0 and z_deg == 0:
            return keypoints

        rot = Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True)
        return keypoints @ rot.as_matrix().T

    def reset_filter(self):
        """Reset low-pass filter state."""
        self.lp_filter.reset()

    def reset(self):
        """Reset all state (filter and optimizer)."""
        self.lp_filter.reset()
        self.optimizer.last_qpos = None

    @property
    def num_joints(self) -> int:
        """Number of joint angles."""
        return self.optimizer.num_joints


__all__ = ["Retargeter"]
