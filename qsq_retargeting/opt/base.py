"""Base classes and utilities for hand retargeting optimizers."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nlopt
import numpy as np
import yaml

from ..robot import RobotWrapper


@dataclass
class TimingStats:
    """Timing statistics for optimizer performance analysis."""
    preprocess_ms: float = 0.0
    fk_ms: float = 0.0
    jacobian_ms: float = 0.0
    gradient_ms: float = 0.0
    nlopt_ms: float = 0.0
    total_ms: float = 0.0
    call_count: int = 0
    iter_counts: List[int] = field(default_factory=list)
    # Per-frame iteration losses: list of lists, each inner list is losses per iteration
    iter_losses: List[List[float]] = field(default_factory=list)
    # Current frame's iteration losses (temporary storage during optimization)
    _current_iter_losses: List[float] = field(default_factory=list)

    def reset(self):
        """Reset all timing statistics."""
        self.preprocess_ms = 0.0
        self.fk_ms = 0.0
        self.jacobian_ms = 0.0
        self.gradient_ms = 0.0
        self.nlopt_ms = 0.0
        self.total_ms = 0.0
        self.call_count = 0
        self.iter_counts = []
        self.iter_losses = []
        self._current_iter_losses = []

    def start_frame(self):
        """Start recording for a new frame."""
        self._current_iter_losses = []

    def record_iter_loss(self, loss: float):
        """Record loss for current iteration."""
        self._current_iter_losses.append(loss)

    def end_frame(self, num_evals: int):
        """End recording for current frame."""
        self.iter_counts.append(num_evals)
        self.iter_losses.append(self._current_iter_losses.copy())
        self._current_iter_losses = []

    def get_last_iter_losses(self) -> List[float]:
        """Get iteration losses for the last frame."""
        if self.iter_losses:
            return self.iter_losses[-1]
        return []

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'preprocess_ms': self.preprocess_ms,
            'fk_ms': self.fk_ms,
            'jacobian_ms': self.jacobian_ms,
            'gradient_ms': self.gradient_ms,
            'nlopt_ms': self.nlopt_ms,
            'total_ms': self.total_ms,
            'call_count': self.call_count,
        }

    def get_avg(self) -> Dict[str, float]:
        """Get average timing per call."""
        if self.call_count == 0:
            return self.to_dict()
        return {
            'preprocess_ms': self.preprocess_ms / self.call_count,
            'fk_ms': self.fk_ms / self.call_count,
            'jacobian_ms': self.jacobian_ms / self.call_count,
            'gradient_ms': self.gradient_ms / self.call_count,
            'nlopt_ms': self.nlopt_ms / self.call_count,
            'total_ms': self.total_ms / self.call_count,
            'call_count': self.call_count,
        }

    def get_iter_stats(self) -> Dict[str, float]:
        """Get iteration count statistics."""
        if not self.iter_counts:
            return {}
        arr = np.array(self.iter_counts)
        return {
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'p90': float(np.percentile(arr, 90)),
            'p99': float(np.percentile(arr, 99)),
        }


# Project root for asset path resolution
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent

# Unit conversion: internal computations use cm
M_TO_CM = 100.0
CM_TO_M = 0.01


def huber_loss_np(x: np.ndarray, delta: float = 2.0) -> np.ndarray:
    """Huber loss function (smooth L1 loss)."""
    abs_x = np.abs(x)
    return np.where(
        abs_x <= delta,
        0.5 * x ** 2,
        delta * (abs_x - 0.5 * delta)
    )


def huber_loss_grad_np(x: np.ndarray, delta: float = 2.0) -> np.ndarray:
    """Gradient of Huber loss w.r.t. x (numpy version)."""
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x, delta * np.sign(x))


class LPFilter:
    """Low-pass filter for smoothing joint positions."""

    def __init__(self, alpha: float):
        """Initialize filter.

        Args:
            alpha: Filter coefficient (0 < alpha <= 1).
                   Smaller = smoother but more latency.
        """
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x: np.ndarray) -> np.ndarray:
        """Apply filter to new value."""
        if not self.is_init:
            self.y = x.copy()
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        """Reset filter state."""
        self.y = None
        self.is_init = False


class BaseOptimizer(ABC):
    """Base class for hand retargeting optimizers.

    All parameters are read from configuration dict (loaded from YAML).
    Supports multiple robot hands (Wuji Hand, Shadow Hand, etc.) via configuration.
    """

    # MediaPipe keypoint indices
    MP_ORIGIN_IDX = 0  # Wrist
    MP_TIP_INDICES = [4, 8, 12, 16, 20]  # Fingertips
    MP_PIP_INDICES = [2, 6, 10, 14, 18]  # PIP joints (thumb uses MCP=2)
    MP_DIP_INDICES = [3, 7, 11, 15, 19]  # DIP joints

    # Default link names for different robot hands
    ROBOT_CONFIGS = {
        # Shadow Hand (MuJoCo Menagerie style, rh_/lh_ prefix) - high quality meshes
        # Uses custom URDF that exactly matches MuJoCo Menagerie joint axes
        'shadow_hand': {
            'origin_link': 'rh_palm',  # Will be lh_palm for left hand
            'tip_links': ['rh_thtip', 'rh_fftip', 'rh_mftip', 'rh_rftip', 'rh_lftip'],
            'link1_names': ['rh_thproximal', 'rh_ffproximal', 'rh_mfproximal', 'rh_rfproximal', 'rh_lfproximal'],
            'link3_names': ['rh_thmiddle', 'rh_ffmiddle', 'rh_mfmiddle', 'rh_rfmiddle', 'rh_lfmiddle'],
            'link4_names': ['rh_thdistal', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal'],
            'urdf_subdir': 'assets/shadow_hand',
            'urdf_file': {'right': 'right_hand_mj.urdf', 'left': 'left_hand_mj.urdf'},
            'mjcf_subdir': 'assets/shadow_hand',
            'mjcf_file': {'right': 'scene_right.xml', 'left': 'scene_left.xml'},
            'num_fingers': 5,
        },
        # Wuji Hand (5 fingers x 4 joints = 20 DOF)
        'wuji_hand': {
            'origin_link': 'right_palm_link',
            'tip_links': ['right_finger1_tip_link', 'right_finger2_tip_link', 'right_finger3_tip_link', 'right_finger4_tip_link', 'right_finger5_tip_link'],
            'link1_names': ['right_finger1_link1', 'right_finger2_link1', 'right_finger3_link1', 'right_finger4_link1', 'right_finger5_link1'],
            'link3_names': ['right_finger1_link3', 'right_finger2_link3', 'right_finger3_link3', 'right_finger4_link3', 'right_finger5_link3'],
            'link4_names': ['right_finger1_link4', 'right_finger2_link4', 'right_finger3_link4', 'right_finger4_link4', 'right_finger5_link4'],
            'num_fingers': 5,
        },
        # Allegro Hand (4 fingers: thumb, index, middle, ring - no pinky)
        # Finger order: thumb (link_12~15), index (link_0~3), middle (link_4~7), ring (link_8~11)
        'allegro_hand': {
            'origin_link': 'base_link',
            'tip_links': ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip'],
            'link1_names': ['link_12.0', 'link_0.0', 'link_4.0', 'link_8.0'],
            'link3_names': ['link_13.0', 'link_1.0', 'link_5.0', 'link_9.0'],
            'link4_names': ['link_14.0', 'link_2.0', 'link_6.0', 'link_10.0'],
            'num_fingers': 4,
        },
        # Inspire Hand (5 fingers, 2-DOF per non-thumb finger)
        # Non-thumb: proximal → intermediate → tip(fixed), so link3=proximal(PIP), link4=intermediate(DIP)
        'inspire_hand': {
            'origin_link': 'hand_base_link',
            'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
            'link1_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
            'link3_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
            'link4_names': ['thumb_intermediate', 'index_intermediate', 'middle_intermediate', 'ring_intermediate', 'pinky_intermediate'],
            'num_fingers': 5,
        },
        # Ability Hand (5 fingers, 2 links each)
        'ability_hand': {
            'origin_link': 'base',
            'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
            'link1_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
            'link3_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
            'link4_names': ['thumb_L2', 'index_L2', 'middle_L2', 'ring_L2', 'pinky_L2'],
            'num_fingers': 5,
        },
        # Leap Hand (4 fingers + thumb, no pinky)
        'leap_hand': {
            'origin_link': 'base',
            'tip_links': ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head'],
            'link1_names': ['thumb_pip', 'pip', 'pip_2', 'pip_3'],
            'link3_names': ['thumb_dip', 'dip', 'dip_2', 'dip_3'],
            'link4_names': ['thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3'],
            'num_fingers': 4,
        },
        # SVH Hand (5 fingers)
        'svh_hand': {
            'origin_link': 'right_hand_base_link',
            'tip_links': ['thtip', 'fftip', 'mftip', 'rftip', 'lftip'],
            'link1_names': ['right_hand_z', 'right_hand_l', 'right_hand_k', 'right_hand_j', 'right_hand_i'],
            'link3_names': ['right_hand_a', 'right_hand_p', 'right_hand_o', 'right_hand_n', 'right_hand_m'],
            'link4_names': ['right_hand_b', 'right_hand_t', 'right_hand_s', 'right_hand_r', 'right_hand_q'],
            'num_fingers': 5,
        },
        # LinkerHand L21
        'linkerhand_l21': {
            'origin_link': 'hand_base_link',
            'tip_links': ['thumb_distal', 'index_middle', 'middle_middle', 'ring_middle', 'pinky_middle'],
            'link1_names': ['thumb_metacarpals', 'index_metacarpals', 'middle_metacarpals', 'ring_metacarpals', 'pinky_metacarpals'],
            'link3_names': ['thumb_metacarpals', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
            'link4_names': ['thumb_distal', 'index_middle', 'middle_middle', 'ring_middle', 'pinky_middle'],
            'link3_offsets': [
                [0.018, 0.000, 0.000],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
            ],
            'tip_offsets': [
                [0.040, 0.000, 0.000],
                [0.000, 0.000, 0.044],
                [0.000, 0.000, 0.044],
                [0.000, 0.000, 0.044],
                [0.000, 0.000, 0.044],
            ],
            'link4_offsets': [
                [0.023, 0.000, 0.000],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
                [0.000, 0.000, 0.022],
            ],
            'urdf_subdir': 'assets/linkerhand_l21',
            'urdf_file': {
                'right': 'right/linkerhand_l21_right_vis.urdf',
                'left': 'left/linkerhand_l21_left_vis.urdf',
            },
            'num_fingers': 5,
            'neutral_qpos': [0.0] * 17,
        },
        # ROHand
        'rohand': {
            'origin_link': 'base_link',
            'tip_links': ['th_distal_link', 'if_distal_link', 'mf_distal_link', 'rf_distal_link', 'lf_distal_link'],
            'link1_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
            'link3_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
            'link4_names': ['th_proximal_link', 'if_proximal_link', 'mf_proximal_link', 'rf_proximal_link', 'lf_proximal_link'],
            'urdf_subdir': 'assets/rohand',
            'urdf_file': {
                'right': 'right/rohand_right_vis.urdf',
                'left': 'left/rohand_left_vis.urdf',
            },
            'num_fingers': 5,
        },
        # Unitree Dex5
        'unitree_dex5_hand': {
            'origin_link': 'base_link00',
            'tip_links': ['Link_14R', 'Link_24R', 'Link_34R', 'Link_44R', 'Link_54R'],
            'link1_names': ['Link_11R', 'Link_21R', 'Link_31R', 'Link_41R', 'Link_51R'],
            'link3_names': ['Link_12R', 'Link_22R', 'Link_32R', 'Link_42R', 'Link_52R'],
            'link4_names': ['Link_13R', 'Link_23R', 'Link_33R', 'Link_43R', 'Link_53R'],
            'urdf_subdir': 'assets/unitree_dex5_hand',
            'urdf_file': {
                'right': 'right/Dex5-URDF-R.urdf',
                'left': 'left/Dex5-URDF-L.urdf',
            },
            'num_fingers': 5,
            'neutral_qpos': [0.0] * 20,
        },
    }

    def __init__(self, config: dict):
        """Initialize optimizer from configuration dict.

        Args:
            config: Configuration dict (typically loaded from YAML)
        """
        self.config = config

        # Extract optimizer config
        opt_config = config.get('optimizer', {})
        self.hand_side = opt_config.get('hand_side', 'right').lower()
        if self.hand_side not in ['right', 'left']:
            raise ValueError(f"hand_side must be 'right' or 'left', got {self.hand_side}")

        # Extract retarget config
        retarget_config = config.get('retarget', {})
        self.huber_delta = retarget_config.get('huber_delta', 2.0)
        self.norm_delta = retarget_config.get('norm_delta', 0.04)

        # Extract robot config
        robot_config = config.get('robot', {})
        robot_type = robot_config.get('type', 'shadow_hand')

        # Get robot-specific defaults
        if robot_type in self.ROBOT_CONFIGS:
            robot_defaults = self.ROBOT_CONFIGS[robot_type]
        else:
            robot_defaults = self.ROBOT_CONFIGS['shadow_hand']

        # Load URDF - support custom path or use default
        urdf_path = robot_config.get('urdf_path')
        if urdf_path:
            # Custom URDF path (absolute or relative to package root)
            urdf_path = Path(urdf_path)
            if not urdf_path.is_absolute():
                urdf_path = _PROJECT_ROOT / urdf_path
            urdf_path = str(urdf_path.resolve())
        else:
            # Default URDF path based on robot type and hand side
            urdf_subdir = robot_defaults['urdf_subdir']
            # Check for custom urdf_file config (e.g., for Menagerie with rh_/lh_ prefixes)
            urdf_file_config = robot_defaults.get('urdf_file')
            if urdf_file_config and isinstance(urdf_file_config, dict):
                urdf_filename = urdf_file_config.get(self.hand_side, f"{self.hand_side}.urdf")
            else:
                urdf_filename = f"{self.hand_side}.urdf"
            urdf_path = str((_PROJECT_ROOT / urdf_subdir / urdf_filename).resolve())

        self.robot = RobotWrapper(urdf_path)
        self.num_joints = self.robot.model.nq

        # Parse mimic joints from URDF
        self._parse_mimic_joints(urdf_path)

        # Setup NLopt optimizer (dimension = number of independent joints)
        self.num_opt_vars = len(self.independent_indices)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.num_opt_vars)
        self.opt.set_maxeval(50)
        self.opt.set_ftol_abs(1e-4)

        # Apply joint limit overrides from config
        lower_bounds = self.robot.joint_limits[:, 0].copy()
        upper_bounds = self.robot.joint_limits[:, 1].copy()
        clamp_config = retarget_config.get('clamp_joint_lower', {})
        if clamp_config:
            for pattern, min_val in clamp_config.items():
                for ji in range(1, self.robot.model.njoints):
                    jname = self.robot.model.names[ji]
                    idx_q = self.robot.model.joints[ji].idx_q
                    nq = self.robot.model.joints[ji].nq
                    if nq > 0 and pattern in jname:
                        if lower_bounds[idx_q] < min_val:
                            lower_bounds[idx_q] = min_val

        self.opt_lower_bounds = lower_bounds
        self.opt_upper_bounds = upper_bounds
        # NLopt bounds only for independent joints
        self.opt.set_lower_bounds(lower_bounds[self.independent_indices].tolist())
        self.opt.set_upper_bounds(upper_bounds[self.independent_indices].tolist())

        # Link names - from config or robot defaults
        self.origin_link_name = robot_config.get('origin_link', robot_defaults['origin_link'])
        self.task_link_names = robot_config.get('tip_links', robot_defaults['tip_links'])
        self.link1_names = robot_config.get('link1_names', robot_defaults['link1_names'])
        self.link3_names = robot_config.get('link3_names', robot_defaults['link3_names'])
        self.link4_names = robot_config.get('link4_names', robot_defaults['link4_names'])
        self.task_offsets = self._resolve_link_offsets(
            robot_config.get('tip_offsets', robot_defaults.get('tip_offsets')),
            len(self.task_link_names),
        )
        self.link3_offsets = self._resolve_link_offsets(
            robot_config.get('link3_offsets', robot_defaults.get('link3_offsets')),
            len(self.link3_names),
        )
        self.link4_offsets = self._resolve_link_offsets(
            robot_config.get('link4_offsets', robot_defaults.get('link4_offsets')),
            len(self.link4_names),
        )
        neutral_qpos = robot_config.get('neutral_qpos', robot_defaults.get('neutral_qpos'))
        self.neutral_qpos = None if neutral_qpos is None else np.asarray(neutral_qpos, dtype=np.float64)

        # Number of fingers (4 for Allegro/Leap, 5 for others)
        self.num_fingers = robot_config.get('num_fingers', robot_defaults.get('num_fingers', 5))

        # For 4-finger hands, map MediaPipe 5-finger indices to 4-finger robot
        # MediaPipe: thumb=0, index=1, middle=2, ring=3, pinky=4
        # 4-finger: thumb=0, index=1, middle=2, ring=3 (pinky ignored)
        if self.num_fingers == 4:
            self.mp_finger_indices = [0, 1, 2, 3]  # Skip pinky
        else:
            self.mp_finger_indices = [0, 1, 2, 3, 4]  # All 5

        # Handle left/right hand prefix for shadow_hand
        # The config uses 'rh_' prefix by default, replace with 'lh_' for left hand
        if robot_type == 'shadow_hand' and self.hand_side == 'left':
            def replace_prefix(name):
                return name.replace('rh_', 'lh_')
            self.origin_link_name = replace_prefix(self.origin_link_name)
            self.task_link_names = [replace_prefix(n) for n in self.task_link_names]
            self.link1_names = [replace_prefix(n) for n in self.link1_names]
            self.link3_names = [replace_prefix(n) for n in self.link3_names]
            self.link4_names = [replace_prefix(n) for n in self.link4_names]
        elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
            def replace_suffix(name):
                if name == 'base_link00':
                    return 'base_link00L'
                return f"{name[:-1]}L" if name.endswith('R') else name
            self.origin_link_name = replace_suffix(self.origin_link_name)
            self.task_link_names = [replace_suffix(n) for n in self.task_link_names]
            self.link1_names = [replace_suffix(n) for n in self.link1_names]
            self.link3_names = [replace_suffix(n) for n in self.link3_names]
            self.link4_names = [replace_suffix(n) for n in self.link4_names]

        # Build link indices
        self._build_link_indices()

        # Store last solution for warm start
        self.last_qpos = None

    @staticmethod
    def _resolve_link_offsets(offsets_config, count: int) -> np.ndarray:
        """Normalize per-link local offsets to shape (count, 3)."""
        offsets = np.zeros((count, 3), dtype=np.float64)
        if offsets_config is None:
            return offsets

        arr = np.asarray(offsets_config, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"Offset must have 3 values, got shape {arr.shape}")
            offsets[:] = arr.reshape(1, 3)
            return offsets
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Offsets must have shape (N, 3), got {arr.shape}")

        n = min(count, arr.shape[0])
        offsets[:n] = arr[:n]
        return offsets

    def _build_link_indices(self):
        """Build link indices for FK computation."""
        self.computed_link_names = []
        self.computed_link_indices = []
        self.computed_link_offsets = []

        def add_point(name: str, offset: np.ndarray | None = None) -> int:
            self.computed_link_names.append(name)
            self.computed_link_indices.append(self.robot.get_link_index(name))
            if offset is None:
                self.computed_link_offsets.append(np.zeros(3, dtype=np.float64))
            else:
                self.computed_link_offsets.append(np.asarray(offset, dtype=np.float64))
            return len(self.computed_link_indices) - 1

        origin_idx = add_point(self.origin_link_name)
        self.origin_indices = [origin_idx for _ in range(self.num_fingers)]
        self.task_indices = [
            add_point(name, offset)
            for name, offset in zip(self.task_link_names, self.task_offsets)
        ]
        self.link3_indices = [
            add_point(name, offset)
            for name, offset in zip(self.link3_names, self.link3_offsets)
        ]
        self.link4_indices = [
            add_point(name, offset)
            for name, offset in zip(self.link4_names, self.link4_offsets)
        ]
        self.computed_link_offsets = np.asarray(self.computed_link_offsets, dtype=np.float64)

    def _parse_mimic_joints(self, urdf_path: str):
        """Parse mimic joint relationships from URDF.

        Sets up:
            self.independent_indices: indices of independent joints in full qpos
            self.mimic_map: dict mapping mimic_qidx -> (source_qidx, multiplier, offset)
            self.has_mimic: whether any mimic joints exist
        """
        # Build joint name -> idx_q mapping from pinocchio model
        joint_name_to_qidx = {}
        for ji in range(1, self.robot.model.njoints):
            jname = self.robot.model.names[ji]
            nq = self.robot.model.joints[ji].nq
            if nq > 0:
                joint_name_to_qidx[jname] = self.robot.model.joints[ji].idx_q

        # Parse URDF XML for mimic tags
        self.mimic_map = {}  # mimic_qidx -> (source_qidx, multiplier, offset)
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for joint_elem in root.iter('joint'):
                mimic_elem = joint_elem.find('mimic')
                if mimic_elem is not None:
                    joint_name = joint_elem.get('name')
                    source_name = mimic_elem.get('joint')
                    multiplier = float(mimic_elem.get('multiplier', '1.0'))
                    offset = float(mimic_elem.get('offset', '0.0'))

                    if joint_name in joint_name_to_qidx and source_name in joint_name_to_qidx:
                        mimic_qidx = joint_name_to_qidx[joint_name]
                        source_qidx = joint_name_to_qidx[source_name]
                        self.mimic_map[mimic_qidx] = (source_qidx, multiplier, offset)
        except (ET.ParseError, FileNotFoundError):
            pass

        # Determine independent joint indices
        mimic_indices = set(self.mimic_map.keys())
        self.independent_indices = np.array(
            [i for i in range(self.num_joints) if i not in mimic_indices],
            dtype=np.int64
        )
        self.has_mimic = len(self.mimic_map) > 0

        if self.has_mimic:
            # Precompute gradient mapping: for each independent joint, which mimic joints depend on it?
            # mimic_dependents[source_qidx] = [(mimic_qidx, multiplier), ...]
            self._mimic_dependents = {}
            for mimic_qidx, (source_qidx, mult, offset) in self.mimic_map.items():
                if source_qidx not in self._mimic_dependents:
                    self._mimic_dependents[source_qidx] = []
                self._mimic_dependents[source_qidx].append((mimic_qidx, mult))

    def expand_to_full_qpos(self, opt_vars: np.ndarray) -> np.ndarray:
        """Expand independent joint values to full qpos with mimic constraints.

        Args:
            opt_vars: (num_opt_vars,) independent joint values

        Returns:
            full_qpos: (num_joints,) full joint vector with mimic values filled in
        """
        if not self.has_mimic:
            return opt_vars.copy()

        full_qpos = np.zeros(self.num_joints, dtype=np.float64)
        full_qpos[self.independent_indices] = opt_vars

        for mimic_qidx, (source_qidx, mult, offset) in self.mimic_map.items():
            full_qpos[mimic_qidx] = full_qpos[source_qidx] * mult + offset

        # Clip mimic joints to their limits
        full_qpos = np.clip(full_qpos, self.opt_lower_bounds, self.opt_upper_bounds)
        return full_qpos

    def map_gradient_to_independent(self, full_grad: np.ndarray) -> np.ndarray:
        """Map full gradient to independent joint gradient using chain rule.

        For mimic joint j with q_j = q_src * mult + offset:
            dL/dq_src += dL/dq_j * mult

        Args:
            full_grad: (num_joints,) gradient w.r.t. full qpos

        Returns:
            opt_grad: (num_opt_vars,) gradient w.r.t. independent joints
        """
        if not self.has_mimic:
            return full_grad.copy()

        # Start with direct gradients for independent joints
        mapped_grad = full_grad.copy()

        # Add chain rule contributions from mimic joints
        for source_qidx, deps in self._mimic_dependents.items():
            for mimic_qidx, mult in deps:
                mapped_grad[source_qidx] += mapped_grad[mimic_qidx] * mult

        return mapped_grad[self.independent_indices]

    @classmethod
    def from_yaml(cls, yaml_path: str, hand_side: str = None) -> "BaseOptimizer":
        """Create optimizer from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file
            hand_side: Optional hand side override ('left' or 'right')

        Returns:
            Optimizer instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override hand_side if provided
        if hand_side is not None:
            if 'optimizer' not in config:
                config['optimizer'] = {}
            config['optimizer']['hand_side'] = hand_side

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: dict) -> "BaseOptimizer":
        """Create optimizer from configuration dict.

        Args:
            config: Configuration dict

        Returns:
            Optimizer instance
        """
        from .adaptive_analytical import AdaptiveOptimizerAnalytical

        opt_type = config.get('optimizer', {}).get('type', 'AdaptiveOptimizerAnalytical')

        if opt_type == 'AdaptiveOptimizerAnalytical':
            return AdaptiveOptimizerAnalytical(config)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    @abstractmethod
    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles.

        Args:
            mediapipe_keypoints: (21, 3) MediaPipe keypoints in wrist frame
            last_qpos: Initial guess for optimization (warm start)

        Returns:
            qpos: (num_joints,) joint angles
        """
        pass

    @abstractmethod
    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles.

        Args:
            qpos: Joint angles
            mediapipe_keypoints: (21, 3) MediaPipe keypoints

        Returns:
            cost: Total loss value
        """
        pass

    # =========================================================================
    # Common helper methods (shared by subclasses)
    # =========================================================================

    def _get_init_qpos(self, last_qpos: Optional[np.ndarray]) -> np.ndarray:
        """Get initial qpos for optimization (independent joints only, clipped).

        Args:
            last_qpos: Optional last qpos from caller (full qpos)

        Returns:
            Initial values for independent joints (num_opt_vars,)
        """
        if last_qpos is not None:
            full_qpos = np.asarray(last_qpos, dtype=np.float64)
        elif self.last_qpos is not None:
            full_qpos = self.last_qpos
        elif self.neutral_qpos is not None:
            full_qpos = self.neutral_qpos
        else:
            full_qpos = (self.opt_lower_bounds + self.opt_upper_bounds) / 2.0

        full_qpos = np.clip(full_qpos, self.opt_lower_bounds, self.opt_upper_bounds)
        return full_qpos[self.independent_indices]

    def _get_reg_qpos(self, last_qpos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Get regularization qpos for norm_delta term (full qpos).

        Args:
            last_qpos: Optional last qpos from caller

        Returns:
            Regularization qpos (full) or None
        """
        if last_qpos is not None:
            return np.asarray(last_qpos, dtype=np.float64)
        elif self.last_qpos is not None:
            return self.last_qpos
        return None

    def _run_optimization(self, objective_fn, init_qpos: np.ndarray) -> np.ndarray:
        """Run NLopt optimization and update last_qpos.

        Args:
            objective_fn: NLopt objective function (operates on independent joints)
            init_qpos: Initial values for independent joints (num_opt_vars,)

        Returns:
            Optimized full qpos (num_joints,)
        """
        self.opt.set_min_objective(objective_fn)
        try:
            opt_result = self.opt.optimize(init_qpos.tolist())
            opt_vars = np.array(opt_result, dtype=np.float64)
        except RuntimeError as e:
            print(f"[{self.__class__.__name__}] Optimization failed: {e}")
            opt_vars = np.array(init_qpos, dtype=np.float64)

        # Expand to full qpos
        full_qpos = self.expand_to_full_qpos(opt_vars)
        self.last_qpos = full_qpos.astype(np.float64)
        return full_qpos.astype(np.float32)

    def _compute_tip_vectors(self, keypoints: np.ndarray, scaling: float = 1.0) -> np.ndarray:
        """Compute wrist->tip vectors.

        Args:
            keypoints: (21, 3) MediaPipe keypoints in meters
            scaling: Global scaling factor

        Returns:
            vectors: (num_fingers, 3) tip vectors in cm
        """
        wrist = keypoints[self.MP_ORIGIN_IDX]
        tip_indices = [self.MP_TIP_INDICES[i] for i in self.mp_finger_indices]
        vectors = np.array([
            keypoints[idx] - wrist for idx in tip_indices
        ]) * scaling * M_TO_CM
        return vectors.astype(np.float64)

    def _compute_tip_dirs(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute DIP->tip direction vectors (normalized).

        Args:
            keypoints: (21, 3) MediaPipe keypoints

        Returns:
            tip_dirs: (num_fingers, 3) normalized direction vectors
        """
        tip_dirs = []
        for fi in self.mp_finger_indices:
            dip_idx = self.MP_DIP_INDICES[fi]
            tip_idx = self.MP_TIP_INDICES[fi]
            dir_vec = keypoints[tip_idx] - keypoints[dip_idx]
            norm = np.linalg.norm(dir_vec)
            tip_dirs.append(dir_vec / (norm + 1e-8))
        return np.array(tip_dirs, dtype=np.float64)

    def _compute_full_hand_vectors(self, keypoints: np.ndarray, scaling: np.ndarray) -> np.ndarray:
        """Compute full hand vectors (wrist->PIP, wrist->DIP, wrist->TIP).

        Args:
            keypoints: (21, 3) MediaPipe keypoints in meters
            scaling: (num_fingers, 3) scaling factors for each finger and segment

        Returns:
            vectors: (num_fingers*3, 3) vectors in cm [PIP*N, DIP*N, TIP*N]
        """
        wrist = keypoints[self.MP_ORIGIN_IDX]
        nf = self.num_fingers

        pip_indices = [self.MP_PIP_INDICES[i] for i in self.mp_finger_indices]
        dip_indices = [self.MP_DIP_INDICES[i] for i in self.mp_finger_indices]
        tip_indices = [self.MP_TIP_INDICES[i] for i in self.mp_finger_indices]

        # wrist -> PIP (N vectors)
        pip_vectors = np.array([
            keypoints[idx] - wrist for idx in pip_indices
        ]) * scaling[:nf, 0:1]

        # wrist -> DIP (N vectors)
        dip_vectors = np.array([
            keypoints[idx] - wrist for idx in dip_indices
        ]) * scaling[:nf, 1:2]

        # wrist -> TIP (N vectors)
        tip_vectors = np.array([
            keypoints[idx] - wrist for idx in tip_indices
        ]) * scaling[:nf, 2:3]

        # Concatenate and convert to cm
        vectors = np.vstack([pip_vectors, dip_vectors, tip_vectors]) * M_TO_CM
        return vectors.astype(np.float64)
