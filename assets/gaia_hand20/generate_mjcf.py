#!/usr/bin/env python3
"""Generate standalone MuJoCo models from the repository-local Gaia Hand20 URDFs.

The source inertias contain rounded zero values that MuJoCo rejects.  This
converter keeps the source masses and uses a conservative diagonal inertia
floor for robust simulation.  Every source visual mesh is emitted exactly once.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

MIN_INERTIA = 1e-7
FINGERS = ("thumb", "index", "middle", "ring", "little")


def _vec(element: ET.Element | None, key: str, default: str) -> str:
    return default if element is None else element.get(key, default)


def _urdf_rpy_to_mjcf_quat(rpy: str) -> str:
    """Convert URDF fixed-axis roll/pitch/yaw to MuJoCo wxyz quaternion.

    Passing URDF RPY directly to MJCF ``euler`` is incorrect because the two
    formats use different Euler rotation conventions.
    """
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll), returned as MuJoCo w x y z.
    quat = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return " ".join(f"{value:.12g}" for value in quat)


def _safe_diaginertia(link: ET.Element) -> tuple[str, str]:
    inertial = link.find("inertial")
    if inertial is None:
        return "0.001", f"{MIN_INERTIA} {MIN_INERTIA} {MIN_INERTIA}"
    mass_element = inertial.find("mass")
    inertia = inertial.find("inertia")
    mass = max(float(mass_element.get("value", "0.001")), 1e-5)
    diagonal = [MIN_INERTIA, MIN_INERTIA, MIN_INERTIA]
    if inertia is not None:
        diagonal = [max(float(inertia.get(name, "0")), MIN_INERTIA)
                    for name in ("ixx", "iyy", "izz")]
        # MuJoCo requires each principal moment not to exceed the sum of the
        # other two. Rounded CAD export values can marginally violate this.
        largest = max(range(3), key=diagonal.__getitem__)
        others = sum(diagonal) - diagonal[largest]
        diagonal[largest] = min(diagonal[largest], max(others * 0.999, MIN_INERTIA))
    return f"{mass:.12g}", " ".join(f"{value:.12g}" for value in diagonal)


def generate(side: str, root_dir: Path) -> Path:
    urdf_path = root_dir / f"gaiahand20_{side}.urdf"
    urdf = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    child_links = {joint.find("child").get("link") for joint in joints}
    root_links = [name for name in links if name not in child_links]
    if root_links != [f"{side}_base_link"]:
        raise ValueError(f"Unexpected URDF roots: {root_links}")

    children: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent").get("link")
        children.setdefault(parent, []).append(joint)

    mj = ET.Element("mujoco", {"model": f"gaiahand20_{side}"})
    ET.SubElement(mj, "compiler", {
        "angle": "radian", "meshdir": ".", "autolimits": "true",
        "inertiafromgeom": "false",
    })
    ET.SubElement(mj, "option", {
        "timestep": "0.002", "gravity": "0 0 0", "integrator": "implicitfast",
    })
    ET.SubElement(mj, "statistic", {"center": "0 0 0.075", "extent": "0.22"})
    # Match the common project scene used by Leap, Inspire, Ability, Allegro,
    # Shadow and SVH: gradient sky, checker floor, and two scene lights.
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "1280"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})

    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", {"damping": "0.08", "armature": "0.0001", "limited": "true"})
    ET.SubElement(default, "geom", {
        "type": "mesh", "contype": "0", "conaffinity": "0", "group": "1", "density": "0",
    })
    ET.SubElement(default, "position", {"kp": "8", "kv": "0.3", "ctrllimited": "true"})

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2",
    })

    mesh_count = 0
    for link in links.values():
        visual_element = link.find("visual")
        if visual_element is None:
            continue
        mesh = visual_element.find("geometry/mesh")
        if mesh is None:
            continue
        ET.SubElement(asset, "mesh", {
            "name": f"{link.get('name')}_mesh",
            "file": mesh.get("filename"),
        })
        mesh_count += 1
    if mesh_count != 21:
        raise ValueError(f"Expected 21 visual meshes, found {mesh_count}")

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true",
    })
    ET.SubElement(worldbody, "light", {
        "pos": "0.5 0.5 1", "dir": "-0.5 -0.5 -1",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "pos": "0 0 -0.001", "size": "0 0 0.05",
        "type": "plane", "material": "groundplane",
    })

    def emit_body(link_name: str, parent_xml: ET.Element, incoming_joint: ET.Element | None = None) -> None:
        attrs = {"name": link_name}
        if incoming_joint is not None:
            origin = incoming_joint.find("origin")
            attrs["pos"] = _vec(origin, "xyz", "0 0 0")
            rpy = _vec(origin, "rpy", "0 0 0")
            if rpy != "0 0 0":
                attrs["quat"] = _urdf_rpy_to_mjcf_quat(rpy)
        body = ET.SubElement(parent_xml, "body", attrs)
        link = links[link_name]

        # Empty fixed fingertip frames need no inertia or geometry.
        visual_element = link.find("visual")
        if incoming_joint is not None and incoming_joint.get("type") != "fixed":
            inertial = link.find("inertial")
            inertial_origin = None if inertial is None else inertial.find("origin")
            mass, diaginertia = _safe_diaginertia(link)
            inertial_attrs = {
                "pos": _vec(inertial_origin, "xyz", "0 0 0"),
                "mass": mass,
                "diaginertia": diaginertia,
            }
            ET.SubElement(body, "inertial", inertial_attrs)
            limit = incoming_joint.find("limit")
            joint_attrs = {
                "name": incoming_joint.get("name"),
                "type": "hinge",
                "axis": _vec(incoming_joint.find("axis"), "xyz", "0 0 1"),
                "range": f"{limit.get('lower')} {limit.get('upper')}",
            }
            ET.SubElement(body, "joint", joint_attrs)

        if visual_element is not None:
            visual_origin = visual_element.find("origin")
            material = visual_element.find("material/color")
            geom_attrs = {
                "name": f"{link_name}_visual",
                "mesh": f"{link_name}_mesh",
                "rgba": "0.7921569 0.8196078 0.9333333 1" if material is None else material.get("rgba"),
            }
            pos = _vec(visual_origin, "xyz", "0 0 0")
            rpy = _vec(visual_origin, "rpy", "0 0 0")
            if pos != "0 0 0":
                geom_attrs["pos"] = pos
            if rpy != "0 0 0":
                geom_attrs["quat"] = _urdf_rpy_to_mjcf_quat(rpy)
            ET.SubElement(body, "geom", geom_attrs)

        for joint in children.get(link_name, []):
            emit_body(joint.find("child").get("link"), body, joint)

    emit_body(root_links[0], worldbody)

    actuator = ET.SubElement(mj, "actuator")
    active_joints = [joint for joint in joints if joint.get("type") != "fixed"]
    for joint in active_joints:
        limit = joint.find("limit")
        ET.SubElement(actuator, "position", {
            "name": f"{joint.get('name')}_actuator",
            "joint": joint.get("name"),
            "ctrlrange": f"{limit.get('lower')} {limit.get('upper')}",
        })
    if len(active_joints) != 20:
        raise ValueError(f"Expected 20 revolute joints, found {len(active_joints)}")

    equality = ET.SubElement(mj, "equality")
    for joint in active_joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        # Keep all 20 actuators for a one-to-one URDF/MuJoCo qpos interface.
        # A soft equality still preserves the mechanical coupling if the model
        # is driven dynamically rather than by direct qpos assignment.
        multiplier = float(mimic.get("multiplier", "1"))
        offset = float(mimic.get("offset", "0"))
        ET.SubElement(equality, "joint", {
            "name": f"{joint.get('name')}_mimic",
            "joint1": joint.get("name"), "joint2": mimic.get("joint"),
            "polycoef": f"{offset:g} {multiplier:g} 0 0 0",
            "solref": "0.004 1",
        })

    output = root_dir / f"gaiahand20_{side}_mujoco.xml"
    tree = ET.ElementTree(mj)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("right", "left", "both"), default="both")
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    sides = ("right", "left") if args.side == "both" else (args.side,)
    for side in sides:
        print(generate(side, root_dir))


if __name__ == "__main__":
    main()
