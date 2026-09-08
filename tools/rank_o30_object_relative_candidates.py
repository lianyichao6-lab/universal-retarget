#!/usr/bin/env python3
"""Rank completed O30 object-relative candidates by mesh contact and collision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.candidates_dir / "o30_object_relative_ranking.json"
    rows = []
    for candidate in sorted(args.candidates_dir.glob("candidate_*")):
        report_path = candidate / "o30_collision_refined_plan.json"
        if not report_path.is_file():
            report_path = candidate / "o30_object_relative" / "o30_collision_refined_plan.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = [float(value) for value in report.get("active_contact_error_after_mm", [])]
        forbidden = float(report.get("max_forbidden_object_penetration_after_mm", float("inf")))
        self_pen = float(report.get("max_self_penetration_after_mm", float("inf")))
        safe = forbidden <= 0.01 and self_pen <= 0.01
        mean_error = sum(errors) / len(errors) if errors else float("inf")
        rows.append({
            "candidate": candidate.name,
            "safe": safe,
            "active_contact_count": len(errors),
            "active_contact_mean_error_mm": mean_error,
            "active_contact_max_error_mm": max(errors, default=float("inf")),
            "forbidden_object_penetration_mm": forbidden,
            "self_penetration_mm": self_pen,
            "trajectory": report.get("trajectory"),
            "report": str(report_path.resolve()),
        })
    rows.sort(key=lambda row: (not row["safe"], -row["active_contact_count"], row["active_contact_mean_error_mm"], row["active_contact_max_error_mm"]))
    payload = {
        "schema_version": 1,
        "simulation_only": True,
        "candidate_count_completed": len(rows),
        "best_candidate": rows[0] if rows else None,
        "ranking": rows,
        "selection_rule": "safe collision result first, then more mesh-near contact fingers, then lower mean/max O30 fingertip target error",
        "limitations": "This ranking does not prove force closure, frictional stability, hidden-surface correctness, camera-to-hand calibration, or physical safety.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Ranked {len(rows)} completed O30 mesh candidates")
    print(f"  output: {output}")
    if rows:
        print(f"  best: {rows[0]['candidate']} (safe={rows[0]['safe']}, mean_error_mm={rows[0]['active_contact_mean_error_mm']:.2f})")


if __name__ == "__main__":
    main()
