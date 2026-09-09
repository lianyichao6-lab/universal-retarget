#!/usr/bin/env python3
"""Apply the strict O30 full-mesh closure gate to a rigid contact plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh

from anydexretarget.o30_mesh_collision import O30MeshCollisionEvaluator
from anydexretarget.o30_scale import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True, help="Metric source mesh in the plan camera frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closure-steps", type=int, default=32)
    parser.add_argument("--max-object-faces", type=int, default=8000)
    args = parser.parse_args()
    with np.load(args.plan, allow_pickle=False) as data:
        plan = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {"qpos_vector_order", "camera_to_o30_rotation", "camera_to_o30_translation", "object_scale_fixed_to_one"}
    missing = required - set(plan)
    if missing:
        raise ValueError("Plan missing: " + ", ".join(sorted(missing)))
    if not bool(plan["object_scale_fixed_to_one"].item()):
        raise ValueError("Refusing to validate an O30 plan that scales the object")
    mesh = trimesh.load_mesh(args.object_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(plan["camera_to_o30_rotation"], dtype=np.float64).T + np.asarray(plan["camera_to_o30_translation"], dtype=np.float64)[None]
    hand_mesh = args.output.with_suffix(".object_in_hand.stl")
    hand_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(hand_mesh)
    evaluator = O30MeshCollisionEvaluator(
        hand_mesh,
        wrist_position_camera=np.zeros(3),
        canonical_basis_row=np.eye(3),
        object_mesh_is_hand_frame=True,
        max_object_faces=args.max_object_faces,
    )
    passed, frames = evaluator.validate_closure(np.asarray(plan["qpos_vector_order"], dtype=np.float64), steps=args.closure_steps)
    payload = {
        "schema": "anydexretarget.o30_mesh_plan_validation/v2",
        "validation_passed": passed,
        "plan": str(args.plan.resolve()),
        "plan_sha256": file_sha256(args.plan),
        "object_mesh": str(args.object_mesh.resolve()),
        "object_mesh_sha256": file_sha256(args.object_mesh),
        "object_scale_fixed_to_one": True,
        "object_mesh_faces": len(mesh.faces),
        "collision_proxy_faces": len(evaluator.collision_mesh.faces),
        "closure_steps": args.closure_steps,
        "final": frames[-1].as_dict(),
        "intermediate_failures": [index for index, frame in enumerate(frames[:-1]) if frame.self_collision_pairs or frame.forbidden_mesh_collisions],
        "intermediate_clearance_warnings": [index for index, frame in enumerate(frames[:-1]) if frame.nonpad_clearance_violations],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"O30 rigid-plan mesh validation: passed={passed}")
    print(f"  report: {args.output}")


if __name__ == "__main__":
    main()
