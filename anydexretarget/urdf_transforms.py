"""Small URDF fixed-joint transform utilities used by deployment tooling."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _vector(attribute: str | None, default: tuple[float, float, float], name: str) -> np.ndarray:
    if attribute is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(attribute, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain three finite values")
    return values


def rpy_to_matrix(rpy: object) -> np.ndarray:
    """Convert URDF fixed-joint roll/pitch/yaw to a rotation matrix."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]], dtype=np.float64
    )


def fixed_joint_transform(joint: ET.Element) -> np.ndarray:
    """Return the child-frame pose expressed in the parent frame."""
    if joint.get("type") != "fixed":
        raise ValueError(f"joint {joint.get('name', '<unnamed>')} is not fixed")
    origin = joint.find("origin")
    xyz = _vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0), "origin xyz")
    rpy = _vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0), "origin rpy")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_to_matrix(rpy)
    result[:3, 3] = xyz
    return result


def find_fixed_joint_transform(urdf_path: str | Path, *, parent_link: str, child_link: str) -> np.ndarray:
    """Find a named parent-to-child fixed joint in a URDF."""
    root = ET.parse(urdf_path).getroot()
    matches = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None and parent.get("link") == parent_link and child.get("link") == child_link:
            matches.append(joint)
    if not matches:
        raise ValueError(f"No joint found for {parent_link!r} -> {child_link!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple joints found for {parent_link!r} -> {child_link!r}")
    return fixed_joint_transform(matches[0])


__all__ = ["find_fixed_joint_transform", "fixed_joint_transform", "rpy_to_matrix"]
