#!/usr/bin/env python3
"""Build or reuse one MuJoCo collision proxy for an object mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_l25_object_relative_scene import prepare_mujoco_mesh_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=180_000)
    args = parser.parse_args()
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)
    if not 0 < args.max_faces < 200_000:
        raise ValueError("--max-faces must be between 1 and 199999")
    output, metadata = prepare_mujoco_mesh_proxy(
        args.mesh, args.output, args.max_faces
    )
    print("MuJoCo object collision proxy ready")
    print(f"  cache: {'hit' if metadata['cache_hit'] else 'miss'}")
    print(
        f"  faces: {metadata['source_faces']} -> {metadata['proxy_faces']}"
    )
    print(f"  proxy: {output}")
    print(f"  metadata: {output.with_suffix(output.suffix + '.json')}")


if __name__ == "__main__":
    main()
