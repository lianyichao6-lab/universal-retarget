#!/usr/bin/env python3
"""Derive reproducible L25 distal-link endpoint candidates from its STL meshes.

The result is a geometric fingertip endpoint in the local distal-link frame.
It is deliberately not presented as a verified hardware contact-pad model.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def load_binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise ValueError(
            f"{path} is not the expected binary STL size "
            f"({len(data)} bytes, expected {expected_size})"
        )
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    offset = 84
    for triangle in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        vertices[triangle * 3 : triangle * 3 + 3] = np.asarray(values[3:12]).reshape(3, 3)
        offset += 50
    return vertices


def geometric_endpoint(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounding box, end-cap centroid, and the +Z endpoint candidate.

    In the L25 URDF, every distal link is reached from the preceding link along
    its local +Z direction.  We use the centroid of the outermost 2 percent of
    that mesh extent to avoid selecting one tessellation outlier.
    """
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    z_band = max(0.00025, 0.02 * (maximum[2] - minimum[2]))
    end_cap = vertices[vertices[:, 2] >= maximum[2] - z_band]
    candidate = np.median(end_cap, axis=0)
    candidate[2] = maximum[2]
    return minimum, maximum, candidate


def format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("assets/linkerhand_l25/right/meshes"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/l25_tip_offset_audit.md"),
    )
    args = parser.parse_args()

    rows: list[tuple[str, Path, np.ndarray, np.ndarray, np.ndarray]] = []
    for finger in FINGERS:
        mesh_path = args.mesh_dir / f"{finger}_distal.STL"
        vertices = load_binary_stl_vertices(mesh_path)
        minimum, maximum, candidate = geometric_endpoint(vertices)
        rows.append((finger, mesh_path, minimum, maximum, candidate))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# L25 Tip Offset Audit",
        "",
        "Generated from the actual L25 `*_distal.STL` meshes. Units are metres, "
        "in each distal link's local frame.",
        "",
        "The L25 URDF connects each distal link from its preceding link along "
        "local `+Z`. The candidate below is the median X/Y of vertices in the "
        "outermost 2% of mesh Z extent, with Z at the terminal mesh surface.",
        "It is a reproducible geometric fingertip endpoint, not a vendor-verified "
        "finger-pad contact point or hardware command mapping.",
        "",
        "| Finger | Mesh | Local min | Local max | Endpoint candidate (`task_offset`) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finger, mesh_path, minimum, maximum, candidate in rows:
        lines.append(
            f"| {finger} | `{mesh_path}` | {format_vector(minimum)} | "
            f"{format_vector(maximum)} | {format_vector(candidate)} |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "These offsets must be used consistently by adaptive (`tip_offsets` in "
            "`robot_configs.py`) and vector (`task_offset` in the five distal-tip "
            "vectors). The next calibration step estimates human-to-robot scales "
            "using these same points.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for finger, _, _, _, candidate in rows:
        print(f"{finger}: {format_vector(candidate)}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
