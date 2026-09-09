#!/usr/bin/env python3
"""Re-rank identical HUG candidates after O30 contact and full-mesh planning."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from anydexretarget.o30_retarget_backend import BACKENDS

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _run(args: list[str]) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])


def _candidates(path: Path, limit: int | None) -> list[dict[str, str]]:
    with (path / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status", "success") == "success"]
    rows.sort(key=lambda row: int(row.get("rank", "999999")))
    return rows if limit is None else rows[:limit]


def _opposition(contact_path: Path) -> tuple[int, bool]:
    with np.load(contact_path, allow_pickle=False) as data:
        active = np.asarray(data["near_surface"], dtype=np.uint8).astype(bool)
        normals = np.asarray(data["surface_normal_camera"], dtype=np.float64)
        names = [str(item).lower() for item in data["finger_names"]]
    thumb = names.index("thumb")
    others = [index for index, enabled in enumerate(active) if enabled and index != thumb]
    opposed = active[thumb] and any(float(np.dot(normals[thumb], normals[index])) <= -0.2 for index in others)
    return int(active.sum()), bool(opposed)


def _score(validation: dict[str, object], refinement: dict[str, object], opposed: bool) -> float:
    final = validation["final"]
    distances = [float(value) * 1000.0 for value in final.get("fingertip_distances_m", {}).values()]
    mean = float(np.mean(distances)) if distances else 1e6
    maximum = float(np.max(distances)) if distances else 1e6
    forbidden = float(refinement.get("max_forbidden_object_penetration_after_mm", 1e6))
    self_pen = float(refinement.get("max_self_penetration_after_mm", 1e6))
    return mean + 0.5 * maximum + forbidden * 100.0 + self_pen * 100.0 + (0.0 if opposed else 50.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="vector")
    parser.add_argument("--scale-profile", type=Path)
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    parser.add_argument("--max-evaluations", type=int, default=180)
    parser.add_argument("--collision-max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--collision-max-iterations", type=int, default=160)
    parser.add_argument("--wrist-translation-limit-mm", type=float, default=30.0)
    parser.add_argument("--wrist-rotation-limit-deg", type=float, default=20.0)
    parser.add_argument("--closure-steps", type=int, default=32)
    parser.add_argument("--strict-top-k", type=int, default=8, help="Run expensive refinement and full-mesh closure only for this many HUG-ranked candidates; 0 means all.")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.candidates_dir.is_dir() or not args.object_mesh.is_file():
        raise FileNotFoundError("--candidates-dir or --object-mesh does not exist")
    if args.scale_profile is not None and not args.scale_profile.is_file():
        raise FileNotFoundError(args.scale_profile)
    if args.strict_top_k < 0:
        parser.error("--strict-top-k must be non-negative")
    if min(args.near_surface_gap_mm, args.max_evaluations, args.collision_max_joint_delta_rad, args.collision_max_iterations, args.wrist_translation_limit_mm, args.wrist_rotation_limit_deg, args.closure_steps) <= 0:
        parser.error("All planning limits must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(_candidates(args.candidates_dir, args.limit)):
        candidate = source["candidate"]
        directory = args.output_dir / candidate
        contact = directory / "contact_plan.npz"
        plan = directory / "o30_object_relative_plan.npz"
        scene = directory / "scene"
        refined = directory / "o30_collision_aware_plan.npz"
        refined_scene = directory / "collision_aware_scene"
        validation = directory / "full_mesh_validation.json"
        row: dict[str, object] = {"candidate": candidate, "seed": source.get("seed", ""), "status": "failed", "backend": args.backend}
        try:
            _run([str(TOOLS / "extract_object_relative_contacts.py"), "--canonical-grasp", str(args.candidates_dir / candidate / "canonical_grasp.npz"), "--object-mesh", str(args.object_mesh), "--near-surface-gap-mm", str(args.near_surface_gap_mm), "--output", str(contact)])
            contacts, opposed = _opposition(contact)
            row["active_contact_fingers"] = contacts
            row["thumb_opposed"] = opposed
            command = [str(TOOLS / "plan_o30_rigid_object_relative_grasp.py"), "--contact-plan", str(contact), "--backend", args.backend, "--max-evaluations", str(args.max_evaluations), "--wrist-translation-limit-mm", str(args.wrist_translation_limit_mm), "--wrist-rotation-limit-deg", str(args.wrist_rotation_limit_deg), "--output", str(plan)]
            if args.scale_profile is not None:
                command.extend(["--scale-profile", str(args.scale_profile)])
            _run(command)
            row.update({"status": "screened", "plan": str(plan.resolve())})
            if args.strict_top_k and source_index >= args.strict_top_k:
                row["screen_reason"] = "outside_strict_top_k"
                rows.append(row)
                print("{}: screened (strict validation skipped)".format(candidate), flush=True)
                continue
            _run([str(TOOLS / "build_o30_object_relative_scene.py"), "--plan", str(plan), "--object-mesh", str(args.object_mesh), "--output-dir", str(scene)])
            _run([str(TOOLS / "refine_o30_collision_aware.py"), "--plan", str(plan), "--scene-xml", str(scene / "o30_object_relative_scene.urdf"), "--max-joint-delta-rad", str(args.collision_max_joint_delta_rad), "--max-iterations", str(args.collision_max_iterations), "--output", str(refined)])
            _run([str(TOOLS / "build_o30_object_relative_scene.py"), "--plan", str(refined), "--object-mesh", str(args.object_mesh), "--output-dir", str(refined_scene)])
            _run([str(TOOLS / "validate_o30_mesh_plan.py"), "--plan", str(refined), "--object-mesh", str(args.object_mesh), "--closure-steps", str(args.closure_steps), "--output", str(validation)])
            validation_data = json.loads(validation.read_text(encoding="utf-8"))
            refinement_data = json.loads(refined.with_suffix(".json").read_text(encoding="utf-8"))
            final = validation_data["final"]
            contacts_final = list(final.get("contact_fingers", []))
            row.update({
                "status": "success",
                "validation_passed": bool(validation_data["validation_passed"]),
                "final_contact_fingers": ",".join(contacts_final),
                "final_contact_count": len(contacts_final),
                "max_forbidden_object_penetration_mm": float(refinement_data["max_forbidden_object_penetration_after_mm"]),
                "max_self_penetration_mm": float(refinement_data["max_self_penetration_after_mm"]),
                "plan": str(refined.resolve()),
                "scene": str(refined_scene.resolve()),
                "validation": str(validation.resolve()),
            })
            row["final_o30_score"] = _score(validation_data, refinement_data, opposed)
            row["recommended"] = bool(row["validation_passed"] and opposed and "thumb" in contacts_final and len(contacts_final) >= 3)
        except Exception as exc:
            row["failure"] = str(exc)
        rows.append(row)
        print("{}: {} recommended={}".format(candidate, row["status"], row.get("recommended", False)), flush=True)

    successful = [row for row in rows if row["status"] == "success"]
    successful.sort(key=lambda row: float(row["final_o30_score"]))
    for index, row in enumerate(successful, start=1):
        row["final_o30_rank"] = index
    recommended = [row for row in successful if row.get("recommended")]
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "o30_final_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "anydexretarget.o30_final_ranking/v1",
        "simulation_only": True,
        "hardware_command_generated": False,
        "backend": args.backend,
        "candidate_count": len(rows),
        "successful_plan_count": len(successful),
        "recommended_plan_count": len(recommended),
        "screened_candidate_count": len([row for row in rows if row["status"] == "screened"]),
        "strict_candidate_count": len(successful),
        "selection_rule": "strict full-mesh closure pass, final thumb-plus-two-or-more-finger contact, thumb opposition, then lower actual mesh clearance and penetration",
        "top_candidates": recommended[:3],
        "recommended_top_candidates": recommended[:3],
        "diagnostic_top_candidates": successful[:3],
    }
    (args.output_dir / "best_o30_candidates.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"O30 final re-ranking complete: {len(successful)}/{len(rows)} planned", flush=True)


if __name__ == "__main__":
    main()
