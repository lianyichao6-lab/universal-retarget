"""Device-specific, three-knot O30 qpos-to-HOP calibration profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .hand_contract import O30_ACTIVE_JOINT_NAMES, O30_ACTIVE_SOURCE_NAMES


@dataclass(frozen=True)
class O30HardwareProfile:
    source_joint_names: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    qpos_knots_rad: np.ndarray
    command_knots_u8: np.ndarray
    device_uid: str
    feedback_tolerance_counts: int = 8

    def validate(self) -> None:
        if self.source_joint_names != O30_ACTIVE_SOURCE_NAMES or self.command_joint_names != O30_ACTIVE_JOINT_NAMES:
            raise ValueError("O30 hardware profile joint order does not match the audited 20-axis contract")
        q = np.asarray(self.qpos_knots_rad, dtype=np.float64)
        u = np.asarray(self.command_knots_u8, dtype=np.float64)
        if q.shape != (20, 3) or u.shape != (20, 3) or not np.isfinite(q).all() or not np.isfinite(u).all():
            raise ValueError("O30 hardware profile requires finite 20 x 3 knot arrays")
        if np.any(np.diff(q, axis=1) <= 0) or np.any(u < 0) or np.any(u > 255):
            raise ValueError("O30 hardware knots must have increasing qpos and u8 values in [0,255]")
        if self.feedback_tolerance_counts <= 0:
            raise ValueError("O30 feedback tolerance must be positive")

    def command_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("O30 hardware mapping requires 20 finite qpos values")
        mapped = [
            np.interp(value, self.qpos_knots_rad[index], self.command_knots_u8[index])
            for index, value in enumerate(values)
        ]
        return np.rint(np.clip(mapped, 0, 255)).astype(np.uint8)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "anydexretarget.o30_hardware_profile/v1",
            "hand_model": "o30", "side": "right", "device_uid": self.device_uid,
            "source_joint_names": list(self.source_joint_names),
            "command_joint_names": list(self.command_joint_names),
            "qpos_knots_rad": np.asarray(self.qpos_knots_rad, dtype=np.float64).tolist(),
            "command_knots_u8": np.asarray(self.command_knots_u8, dtype=np.int64).tolist(),
            "feedback_tolerance_counts": self.feedback_tolerance_counts,
        }

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "O30HardwareProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "anydexretarget.o30_hardware_profile/v1":
            raise ValueError(f"Not an O30 hardware profile: {path}")
        profile = cls(
            source_joint_names=tuple(payload["source_joint_names"]),
            command_joint_names=tuple(payload["command_joint_names"]),
            qpos_knots_rad=np.asarray(payload["qpos_knots_rad"], dtype=np.float64),
            command_knots_u8=np.asarray(payload["command_knots_u8"], dtype=np.float64),
            device_uid=str(payload.get("device_uid", "unbound")),
            feedback_tolerance_counts=int(payload.get("feedback_tolerance_counts", 8)),
        )
        profile.validate()
        return profile


def nominal_profile(model: mujoco.MjModel, *, device_uid: str = "unbound") -> O30HardwareProfile:
    by_name = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
        for index in range(model.njnt)
    }
    knots = []
    for name in O30_ACTIVE_SOURCE_NAMES:
        index = by_name.get(name)
        if index is None:
            raise ValueError(f"O30 model lacks {name}")
        lower, upper = model.jnt_range[index]
        knots.append([lower, 0.5 * (lower + upper), upper])
    return O30HardwareProfile(
        source_joint_names=O30_ACTIVE_SOURCE_NAMES,
        command_joint_names=O30_ACTIVE_JOINT_NAMES,
        qpos_knots_rad=np.asarray(knots, dtype=np.float64),
        command_knots_u8=np.tile(np.asarray([0, 128, 255], dtype=np.float64), (20, 1)),
        device_uid=device_uid,
    )


__all__ = ["O30HardwareProfile", "nominal_profile"]
