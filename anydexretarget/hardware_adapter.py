"""Dry-run LinkerHand G20 qpos to SDK command adapter.

This module never opens a device or sends a command. It only implements the
verified didongtai joint-name/order/range conversion for offline inspection.

G20 uses 20 SDK channels (indices 0-19). The joint mapping is identical to
L25 channels 0-19; G20 simply has no tip-mirror channels (L25's 20-24).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G20_QPOS_JOINTS = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
)

INDEPENDENT_TO_SDK = {
    "thumb_cmc_pitch": 0, "index_mcp_pitch": 1, "middle_mcp_pitch": 2,
    "ring_mcp_pitch": 3, "pinky_mcp_pitch": 4, "thumb_cmc_yaw": 10,
    "index_mcp_roll": 6, "middle_mcp_roll": 7, "ring_mcp_roll": 8,
    "pinky_mcp_roll": 9, "thumb_cmc_roll": 5, "thumb_mcp": 15,
    "index_pip": 16, "middle_pip": 17, "ring_pip": 18, "pinky_pip": 19,
}
# (qpos lower, qpos upper, hardware arc lower, hardware arc upper).
RANGES = {
    0: (0.0, .83, 0.0, .9), 1: (0.0, 1.22, 0.0, 1.57), 2: (0.0, 1.22, 0.0, 1.57),
    3: (0.0, 1.22, 0.0, 1.57), 4: (0.0, 1.22, 0.0, 1.57), 5: (0.0, 1.39, 0.0, 1.3),
    6: (-.23, .23, -.26, .26), 7: (-.23, .23, -.26, .26), 8: (-.23, .23, -.26, .26),
    9: (-.23, .23, -.26, .26), 10: (0.0, 1.57, -.26, .61), 15: (0.0, 1.25, 0.0, 1.57),
    16: (0.0, 1.75, 0.0, 1.57), 17: (0.0, 1.75, 0.0, 1.57), 18: (0.0, 1.75, 0.0, 1.57),
    19: (0.0, 1.75, 0.0, 1.57),
}
INVERTED_SDK = {0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19}

@dataclass(frozen=True)
class G20DryRunCommand:
    values: np.ndarray
    joint_names: tuple[str, ...]
    hardware: str = "linkerhand_g20"
    dry_run: bool = True

class G20HardwareAdapter:
    """Convert named G20 qpos to 20 integer SDK channels without sending."""
    def qpos_to_command(self, qpos: np.ndarray, joint_names=G20_QPOS_JOINTS) -> G20DryRunCommand:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (21,) or not np.isfinite(values).all():
            raise ValueError("G20 qpos must be finite with shape (21,)")
        by_name = {str(name): float(value) for name, value in zip(joint_names, values)}
        if set(G20_QPOS_JOINTS) - by_name.keys():
            raise ValueError("qpos joint names do not cover the G20 model")
        sdk = np.zeros(20, dtype=np.int64)
        for name, sdk_idx in INDEPENDENT_TO_SDK.items():
            q = float(by_name[name])
            sim_min, sim_max, hw_min, hw_max = RANGES[sdk_idx]
            q = float(np.clip(q, sim_min, sim_max))
            arc = (q - sim_min) * (hw_max - hw_min) / (sim_max - sim_min) + hw_min
            fraction = (arc - hw_min) / (hw_max - hw_min)
            sdk[sdk_idx] = int(np.rint(np.clip(255.0 * (1.0 - fraction if sdk_idx in INVERTED_SDK else fraction), 0.0, 255.0)))
        return G20DryRunCommand(sdk, tuple(f"sdk_{i}" for i in range(20)))

# Backward compatibility aliases
L25_QPOS_JOINTS = G20_QPOS_JOINTS
L25HardwareAdapter = G20HardwareAdapter
L25DryRunCommand = G20DryRunCommand

__all__ = [
    "G20HardwareAdapter", "G20DryRunCommand", "G20_QPOS_JOINTS",
    "L25HardwareAdapter", "L25DryRunCommand", "L25_QPOS_JOINTS",
]
