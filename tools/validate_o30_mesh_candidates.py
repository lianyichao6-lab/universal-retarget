#!/usr/bin/env python3
"""Strictly validate O30 HUG candidates against a complete object mesh."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from anydexretarget.o30_mesh_collision import O30MeshCollisionEvaluator
from anydexretarget.o30_scale import O30ScaleProfile, file_sha256


def _qpos(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    value = np.asarray(records[-1]["target"], dtype=np.float64)
    if value.shape != (20,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid O30 trajectory: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True, help="Metric mesh in the anchor camera frame")
    parser.add_argument("--scale-profile", type=Path, required=True)
    parser.add_argument("--closure-steps", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.closure_steps < 2:
        parser.error("--closure-steps must be at least 2")
    profile = O30ScaleProfile.load(args.scale_profile)
    rows = []
    for directory in sorted(path for path in args.candidates_dir.glob("candidate_*") if path.is_dir()):
        with np.load(directory / "canonical_grasp.npz", allow_pickle=False) as data:
            evaluator = O30MeshCollisionEvaluator(
                args.object_mesh,
                wrist_position_camera=data["wrist_position_camera"],
                canonical_basis_row=data["canonical_basis_row"],
                scale_profile=profile,
            )
        passed, frames = evaluator.validate_closure(_qpos(directory / "trajectory.pkl"), steps=args.closure_steps)
        final = frames[-1]
        row = {
            "candidate": directory.name,
            "validation_passed": passed,
            "final": final.as_dict(),
            "closure_steps": args.closure_steps,
            "failure_frame_count": sum(not frame.passed for frame in frames[:-1]),
        }
        (directory / "o30_full_mesh_validation.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"{directory.name}: passed={passed} contacts={','.join(final.contact_fingers) or '-'}")
    rows.sort(key=lambda item: (
        not bool(item["validation_passed"]),
        -len(item["final"]["contact_fingers"]),
        len(item["final"]["forbidden_mesh_collisions"]),
        len(item["final"]["self_collision_pairs"]),
    ))
    output = args.output or args.candidates_dir / "o30_full_mesh_ranking.json"
    payload = {
        "schema": "anydexretarget.o30_full_mesh_ranking/v1",
        "object_mesh": str(args.object_mesh.resolve()),
        "object_mesh_sha256": file_sha256(args.object_mesh),
        "scale_profile": str(args.scale_profile.resolve()),
        "candidate_count": len(rows),
        "validated_candidates": rows,
        "best_candidate": next((row for row in rows if row["validation_passed"]), None),
        "selection": "thumb plus at least two other fingertip pads; no forbidden full-mesh or cross-finger collision over the complete closure path",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"full-mesh O30 ranking: {output}")


if __name__ == "__main__":
    main()
