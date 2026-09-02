"""Minimal tactile contract for simulated and real fingertip feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COUNT = len(FINGER_NAMES)


@dataclass(frozen=True)
class TactileState:
    """One synchronized five-finger contact/wrench sample."""

    timestamp_s: float
    contact: np.ndarray
    wrenches: np.ndarray

    def __post_init__(self) -> None:
        contact = np.asarray(self.contact, dtype=bool)
        wrenches = np.asarray(self.wrenches, dtype=np.float32)
        if contact.shape != (FINGER_COUNT,):
            raise ValueError(f"contact must have shape ({FINGER_COUNT},)")
        if wrenches.shape != (FINGER_COUNT, 6):
            raise ValueError(f"wrenches must have shape ({FINGER_COUNT}, 6)")
        if not np.isfinite(wrenches).all():
            raise ValueError("wrenches must contain only finite values")
        object.__setattr__(self, "contact", contact.copy())
        object.__setattr__(self, "wrenches", wrenches.copy())
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))

    @property
    def contact_count(self) -> int:
        return int(np.count_nonzero(self.contact))

    def as_observation(self) -> dict[str, np.ndarray | float]:
        return {"timestamp_s": self.timestamp_s, "contact": self.contact.copy(), "wrenches": self.wrenches.copy()}


def tactile_state_from_wrenches(
    wrenches: np.ndarray,
    *,
    timestamp_s: float = 0.0,
    contact_force_threshold: float = 0.1,
) -> TactileState:
    """Build contact flags from per-finger [Fx,Fy,Fz,Tx,Ty,Tz] wrenches."""
    values = np.asarray(wrenches, dtype=np.float32)
    if values.shape != (FINGER_COUNT, 6):
        raise ValueError(f"wrenches must have shape ({FINGER_COUNT}, 6)")
    if not np.isfinite(values).all():
        raise ValueError("wrenches must contain only finite values")
    if not np.isfinite(contact_force_threshold) or contact_force_threshold < 0:
        raise ValueError("contact_force_threshold must be finite and non-negative")
    contact = np.linalg.norm(values[:, :3], axis=1) > float(contact_force_threshold)
    return TactileState(timestamp_s, contact, values)


__all__ = ["FINGER_NAMES", "FINGER_COUNT", "TactileState", "tactile_state_from_wrenches"]
