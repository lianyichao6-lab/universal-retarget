#!/usr/bin/env python3
"""Rank completed O30 mesh-contact trials using actual displayed-mesh distance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _candidate_name(path: Path) -> str:
    return next(part for part in path.parts if part.startswith("candidate_"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.candidates_dir / "o30_object_relative_ranking.json"
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for report_path in sorted(args.candidates_dir.glob("candidate_*/**/o30*_mesh_closed_plan.json")):
        if report_path in seen:
            continue
        seen.add(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mesh_errors = [float(value) for value in report.get("active_mesh_distance_after_mm", [])]
        target_errors = [float(value) for value in report.get("active_contact_error_after_mm", [])]
        forbidden = float(report.get("max_forbidden_object_penetration_after_mm", float("inf")))
        self_pen = float(report.get("max_self_penetration_after_mm", float("inf")))
        safe = forbidden <= 0.01 and self_pen <= 0.01
        try:
            candidate = _candidate_name(report_path.relative_to(args.candidates_dir))
        except StopIteration:
            continue
        trial = str(report_path.parent.relative_to(args.candidates_dir / candidate))
        rows.append({
            "candidate": candidate,
            "trial": trial,
            "safe": safe,
            "active_contact_count": len(mesh_errors),
            "meets_min_contact_count": len(mesh_errors) >= 3,
            "active_mesh_mean_distance_mm": sum(mesh_errors) / len(mesh_errors) if mesh_errors else float("inf"),
            "active_mesh_max_distance_mm": max(mesh_errors, default=float("inf")),
            "active_target_mean_error_mm": sum(target_errors) / len(target_errors) if target_errors else float("inf"),
            "forbidden_object_penetration_mm": forbidden,
            "self_penetration_mm": self_pen,
            "trajectory": report.get("trajectory"),
            "report": str(report_path.resolve()),
        })
    rows.sort(key=lambda row: (
        not bool(row["safe"]),
        not bool(row["meets_min_contact_count"]),
        float(row["active_mesh_mean_distance_mm"]),
        float(row["active_mesh_max_distance_mm"]),
        float(row["active_target_mean_error_mm"]),
    ))
    payload = {
        "schema_version": 2,
        "simulation_only": True,
        "trial_count_completed": len(rows),
        "best_trial": rows[0] if rows else None,
        "ranking": rows,
        "selection_rule": "no forbidden object/cross-finger penetration, then require at least three active contacts, then lower O30 fingertip-to-actual-displayed-mesh distance",
        "limitations": "This ranking does not prove force closure, frictional stability, hidden-surface correctness, camera-to-hand calibration, or physical safety.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Ranked {len(rows)} completed O30 mesh-contact trials")
    print(f"  output: {output}")
    if rows:
        print(f"  best: {rows[0]['candidate']}/{rows[0]['trial']} (safe={rows[0]['safe']}, mesh_mean_mm={rows[0]['active_mesh_mean_distance_mm']:.3f})")


if __name__ == "__main__":
    main()
