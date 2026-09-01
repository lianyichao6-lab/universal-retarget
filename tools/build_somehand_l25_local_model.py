#!/usr/bin/env python3
"""Create a SomeHand-compatible L25 MJCF from the local authoritative model."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
DEFAULT_OUTPUT = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_somehand.xml"
TIP_SITES = {
    "thumb_distal": ("thumb_distal_tip", "-0.008849 -0.000018 0.030758"),
    "index_distal": ("index_distal_tip", "-0.015799 -0.000013 0.022931"),
    "middle_distal": ("middle_distal_tip", "-0.015799 -0.000013 0.022931"),
    "ring_distal": ("ring_distal_tip", "-0.015799 -0.000013 0.022931"),
    "pinky_distal": ("pinky_distal_tip", "-0.015799 -0.000013 0.022931"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    tree = ET.parse(source)
    bodies = {body.attrib.get("name"): body for body in tree.getroot().iter("body")}
    for body_name, (site_name, position) in TIP_SITES.items():
        body = bodies.get(body_name)
        if body is None:
            raise ValueError(f"Local L25 model has no body {body_name!r}")
        if any(site.attrib.get("name") == site_name for site in body.findall("site")):
            raise ValueError(f"Local L25 model already contains site {site_name!r}")
        ET.SubElement(body, "site", name=site_name, pos=position, size="0.004", rgba="1 0 0 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    missing = [name for name, _ in TIP_SITES.values() if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0]
    if missing:
        raise RuntimeError(f"Generated model is missing sites: {missing}")
    print(f"Generated {output}")
    print(f"  source: {source}")
    print(f"  nq={model.nq}, nu={model.nu}, sites added={len(TIP_SITES)}")


if __name__ == "__main__":
    main()
