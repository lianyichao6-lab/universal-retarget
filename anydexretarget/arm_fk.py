"""Forward kinematics for the Luban AR5-5_08 right arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np


AR5_RIGHT_JOINT_NAMES = tuple(f"r_joint_{index}" for index in range(1, 8))
DEFAULT_AR5_FLANGE_FRAME = "AR5-5_08R-W4C4A6-ZY2_flan_link"


class AR5ForwardKinematics:
    """Pinocchio-backed FK with the Luban controller joint-name contract."""

    def __init__(self, urdf_path: str | Path, *, frame_name: str = DEFAULT_AR5_FLANGE_FRAME) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(frame_name)
        if self.frame_id >= self.model.nframes:
            raise ValueError(f"AR5 URDF has no frame {frame_name!r}")
        self.frame_name = frame_name
        self._joint_ids = []
        for name in AR5_RIGHT_JOINT_NAMES:
            candidates = [joint_id for joint_id, model_name in enumerate(self.model.names) if model_name.endswith(f"_joint_{name.rsplit('_', 1)[-1]}")]
            if len(candidates) != 1:
                raise ValueError(f"AR5 URDF does not uniquely contain {name}")
            self._joint_ids.append(candidates[0])
        if self.model.nq != len(AR5_RIGHT_JOINT_NAMES):
            raise ValueError(f"Expected 7 AR5 position DoF, got {self.model.nq}")

    def flange_transform(self, positions: object) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("AR5 positions must be finite with shape (7,)")
        self._pin.forwardKinematics(self.model, self.data, values)
        self._pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = placement.rotation
        result[:3, 3] = placement.translation
        return result

    def flange_transforms(self, positions: object) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError("AR5 positions must be finite with shape (N, 7)")
        return np.asarray([self.flange_transform(row) for row in values])


def reorder_ar5_positions(joint_names: object, positions: object) -> np.ndarray:
    """Reorder a trajectory to ``r_joint_1`` ... ``r_joint_7``."""
    names = [str(name) for name in np.asarray(joint_names).reshape(-1)]
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(names):
        raise ValueError("positions must have shape (N, len(joint_names))")
    if set(names) != set(AR5_RIGHT_JOINT_NAMES):
        raise ValueError(f"joint names must be exactly {AR5_RIGHT_JOINT_NAMES}")
    indices = [names.index(name) for name in AR5_RIGHT_JOINT_NAMES]
    result = values[:, indices]
    if not np.isfinite(result).all():
        raise ValueError("positions must contain only finite values")
    return result


__all__ = ["AR5ForwardKinematics", "AR5_RIGHT_JOINT_NAMES", "DEFAULT_AR5_FLANGE_FRAME", "reorder_ar5_positions"]
