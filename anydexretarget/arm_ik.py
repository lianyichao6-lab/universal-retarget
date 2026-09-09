"""Bounded numerical inverse kinematics for the Luban AR5 right arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arm_fk import AR5ForwardKinematics, DEFAULT_AR5_FLANGE_FRAME
from .luban_arm import homogeneous_transform


@dataclass(frozen=True)
class AR5IkResult:
    """Result of a pose-constrained AR5 inverse-kinematics solve."""

    positions: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class AR5NumericalIK:
    """Solve a flange pose with the AR5 URDF limits enforced by SciPy."""

    def __init__(self, urdf_path: str | Path, *, frame_name: str = DEFAULT_AR5_FLANGE_FRAME) -> None:
        self.fk = AR5ForwardKinematics(urdf_path, frame_name=frame_name)
        lower = np.asarray(self.fk.model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self.fk.model.upperPositionLimit, dtype=np.float64)
        if lower.shape != (7,) or upper.shape != (7,) or not np.all(lower < upper):
            raise ValueError("AR5 URDF must define valid limits for seven joints")
        self.lower = lower
        self.upper = upper

    def solve(
        self,
        target_transform: object,
        seed_positions: object,
        *,
        position_tolerance_m: float = 0.002,
        orientation_tolerance_rad: float = 0.0524,
        max_nfev: int = 300,
    ) -> AR5IkResult:
        """Return a joint-limited solution or fail when the target is not reached."""
        from scipy.optimize import least_squares

        target = homogeneous_transform(target_transform, "target_transform")
        seed = np.asarray(seed_positions, dtype=np.float64)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("seed_positions must contain seven finite values")
        if position_tolerance_m <= 0.0 or orientation_tolerance_rad <= 0.0 or max_nfev < 1:
            raise ValueError("IK tolerances and max_nfev must be positive")
        initial = np.clip(seed, self.lower + 1e-8, self.upper - 1e-8)

        def residual(positions: np.ndarray) -> np.ndarray:
            current = self.fk.flange_transform(positions)
            position_error = (current[:3, 3] - target[:3, 3]) * 10.0
            orientation_error = self.fk._pin.log3(current[:3, :3].T @ target[:3, :3])
            return np.concatenate((position_error, orientation_error))

        solution = least_squares(
            residual,
            initial,
            bounds=(self.lower, self.upper),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        final_transform = self.fk.flange_transform(solution.x)
        position_error = float(np.linalg.norm(final_transform[:3, 3] - target[:3, 3]))
        orientation_error = float(
            np.linalg.norm(self.fk._pin.log3(final_transform[:3, :3].T @ target[:3, :3]))
        )
        # SciPy may hit max_nfev after already entering the explicit robot
        # tolerances. The geometric tolerances, not its generic termination
        # status, define whether this bounded IK target is executable.
        if position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad:
            raise RuntimeError(
                "AR5 numerical IK failed: "
                f"status={solution.status}, position_error_m={position_error:.6f}, "
                f"orientation_error_rad={orientation_error:.6f}"
            )
        return AR5IkResult(
            positions=np.asarray(solution.x, dtype=np.float64),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            iterations=int(solution.nfev),
        )


__all__ = ["AR5IkResult", "AR5NumericalIK"]
