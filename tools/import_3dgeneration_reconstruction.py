#!/usr/bin/env python3
"""Import a 3DGenerationPipeline mesh into the common AnyDex grasp contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from anydexretarget.reconstruction_adapter import adapt_reconstruction_mesh, load_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Metric GLB/OBJ/PLY exported by 3DGenerationPipeline.")
    parser.add_argument("--anchor-pointcloud", type=Path, required=True, help="Masked Gemini RGB-D cloud in anchor optical frame.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="3dgenerationpipeline-fine")
    parser.add_argument("--anchor-frame", default="hand_camera_color_optical_frame")
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--transform", type=Path, help="Measured 4x4 T_anchor_object (.npy/.npz/.json/.txt).")
    alignment.add_argument("--auto-align", action="store_true", help="Estimate rigid pose from the partial RGB-D surface; inspect the overlay.")
    parser.add_argument("--source-unit", choices=("m", "cm", "mm"), default="m")
    parser.add_argument("--known-max-dimension-mm", type=float, help="Override mesh scale using the object's measured OBB maximum edge.")
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--alignment-samples", type=int, default=4000)
    parser.add_argument("--merge-radius-mm", type=float, default=4.0)
    parser.add_argument("--completed-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unit_scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[args.source_unit]
    result = adapt_reconstruction_mesh(
        mesh_path=args.mesh,
        anchor_cloud_path=args.anchor_pointcloud,
        output_dir=args.output_dir,
        backend=args.backend,
        anchor_frame=args.anchor_frame,
        transform=None if args.transform is None else load_transform(args.transform),
        auto_align=args.auto_align,
        source_unit_scale=unit_scale,
        known_max_dimension_m=None if args.known_max_dimension_mm is None else args.known_max_dimension_mm / 1000.0,
        surface_samples=args.surface_samples,
        alignment_samples=args.alignment_samples,
        merge_radius_m=args.merge_radius_mm / 1000.0,
        completed_confidence=args.completed_confidence,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    outputs = result["outputs"]
    metadata = result["metadata"]
    print("3DGenerationPipeline reconstruction imported")
    print(f"  alignment: {metadata['alignment']['mode']}")
    print(f"  visible RMSE: {metadata['alignment']['visible_surface_rmse_m'] * 1000:.2f} mm")
    print(f"  measured/completed: {metadata['visible_measured_points']}/{metadata['completed_mesh_points']}")
    print(f"  mesh: {outputs['mesh']}")
    print(f"  HUG surface: {outputs['surface']}")
    print(f"  metadata: {outputs['metadata']}")


if __name__ == "__main__":
    main()
