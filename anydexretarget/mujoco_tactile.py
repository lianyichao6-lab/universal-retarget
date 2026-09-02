"""MuJoCo contact extraction for the five-finger tactile contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .tactile_adapter import FINGER_NAMES, TactileState, tactile_state_from_wrenches


def fingertip_geom_ids(model: Any, *, suffix: str = "_distal_visual") -> dict[str, int]:
    """Resolve the L25 fingertip geom ids by the stable finger naming scheme."""
    ids: dict[str, int] = {}
    for finger in FINGER_NAMES:
        name = f"{finger}{suffix}"
        try:
            geom_id = int(model.geom(name).id)
        except (KeyError, AttributeError):
            geom_id = -1
        if geom_id < 0:
            raise ValueError(f"MuJoCo model is missing fingertip geom {name!r}")
        ids[finger] = geom_id
    return ids


def tactile_state_from_mujoco(
    model: Any,
    data: Any,
    fingertip_ids: Mapping[str, int],
    *,
    object_geom_ids: Sequence[int] | None = None,
    timestamp_s: float = 0.0,
    contact_force_threshold: float = 0.1,
) -> TactileState:
    """Aggregate MuJoCo contact wrenches into one sample per fingertip.

    ``mj_contactForce`` reports a six-vector in the contact frame.  The
    tactile contract intentionally keeps that frame, so real sensors can use
    the same field layout without pretending the simulation has a sensor frame
    calibration that it does not have.
    """
    missing = [finger for finger in FINGER_NAMES if finger not in fingertip_ids]
    if missing:
        raise ValueError(f"fingertip_ids is missing {missing}")
    finger_by_geom = {int(geom_id): index for index, geom_id in enumerate(fingertip_ids.values())}
    allowed_objects = None if object_geom_ids is None else {int(geom_id) for geom_id in object_geom_ids}
    wrenches = np.zeros((len(FINGER_NAMES), 6), dtype=np.float32)
    force = np.zeros(6, dtype=np.float64)

    import mujoco

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        finger_index = finger_by_geom.get(geom1)
        other_geom = geom2
        if finger_index is None:
            finger_index = finger_by_geom.get(geom2)
            other_geom = geom1
        if finger_index is None:
            continue
        if allowed_objects is not None and other_geom not in allowed_objects:
            continue
        force.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_index, force)
        wrenches[finger_index] += force.astype(np.float32)

    return tactile_state_from_wrenches(
        wrenches,
        timestamp_s=timestamp_s,
        contact_force_threshold=contact_force_threshold,
    )


__all__ = ["fingertip_geom_ids", "tactile_state_from_mujoco"]
