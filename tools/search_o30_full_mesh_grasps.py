#!/usr/bin/env python3
"""Run the O30 layered search: HUG candidates -> pad patterns -> mesh gate.

This is offline only.  It uses the no-object-scale rigid planner and exports a
hardware bundle only after the strict full-mesh closure validator passes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _run(arguments: list[str]) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, *arguments], check=True, env=env)


def _candidates(path: Path, limit: int) -> list[Path]:
    csv_path = path / "candidates.csv"
    if not csv_path.is_file():
        return sorted(item for item in path.glob("candidate_*") if item.is_dir())[:limit]
    with csv_path.open(encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row.get("rank", 999999)))
    return [path / row["candidate"] for row in rows[:limit] if (path / row["candidate"]).is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--scale-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--wrist-translation-limit-mm", type=float, default=30.0)
    parser.add_argument("--wrist-rotation-limit-deg", type=float, default=20.0)
    args = parser.parse_args()
    if args.max_candidates <= 0 or args.wrist_translation_limit_mm <= 0 or args.wrist_rotation_limit_deg <= 0:
        parser.error("search limits must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    patterns = [
        ("thumb", *others)
        for count in (2, 3, 4)
        for others in itertools.combinations(FINGERS[1:], count)
    ]
    for candidate in _candidates(args.candidates_dir, args.max_candidates):
        contact_plan = args.output_dir / candidate.name / "contact_plan.npz"
        _run([str(TOOLS / "extract_object_relative_contacts.py"), "--canonical-grasp", str(candidate / "canonical_grasp.npz"), "--object-mesh", str(args.object_mesh), "--output", str(contact_plan)])
        for pattern in patterns:
            label = "_".join(pattern)
            trial = args.output_dir / candidate.name / label
            plan = trial / "o30_rigid_plan.npz"
            validation = trial / "full_mesh_validation.json"
            _run([
                str(TOOLS / "plan_o30_rigid_object_relative_grasp.py"), "--contact-plan", str(contact_plan),
                "--scale-profile", str(args.scale_profile), "--contact-fingers", ",".join(pattern),
                "--wrist-translation-limit-mm", str(args.wrist_translation_limit_mm),
                "--wrist-rotation-limit-deg", str(args.wrist_rotation_limit_deg), "--output", str(plan),
            ])
            _run([str(TOOLS / "validate_o30_mesh_plan.py"), "--plan", str(plan), "--object-mesh", str(args.object_mesh), "--output", str(validation)])
            report = json.loads(validation.read_text(encoding="utf-8"))
            rows.append({"candidate": candidate.name, "pattern": label, "plan": str(plan), "validation": str(validation), "passed": bool(report["validation_passed"]), "contact_count": len(report["final"]["contact_fingers"]), "forbidden": len(report["final"]["forbidden_mesh_collisions"]), "self_collision": len(report["final"]["self_collision_pairs"])})
    rows.sort(key=lambda item: (not item["passed"], -item["contact_count"], item["forbidden"], item["self_collision"]))
    passed = next((item for item in rows if item["passed"]), None)
    summary = {"schema": "anydexretarget.o30_layered_search/v1", "object_mesh": str(args.object_mesh.resolve()), "scale_profile": str(args.scale_profile.resolve()), "evaluated_trials": rows, "best": passed, "selection": "top HUG candidates -> thumb-plus-two-or-more-finger patterns -> no-scale rigid fit -> strict full-mesh closure gate"}
    summary_path = args.output_dir / "o30_layered_search.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if passed is None:
        raise RuntimeError(f"No O30 trial passed strict full-mesh validation; see {summary_path}")
    final_scene = args.output_dir / "best_scene"
    _run([str(TOOLS / "build_o30_object_relative_scene.py"), "--plan", passed["plan"], "--object-mesh", str(args.object_mesh), "--output-dir", str(final_scene)])
    _run([str(TOOLS / "export_o30_validated_plan_bundle.py"), "--plan", passed["plan"], "--validation", passed["validation"], "--output", str(args.output_dir / "o30_grasp_bundle.npz")])
    print(f"O30 layered search passed: {passed['candidate']}/{passed['pattern']}")
    print(f"  bundle: {args.output_dir / 'o30_grasp_bundle.npz'}")


if __name__ == "__main__":
    main()
