#!/usr/bin/env python3
"""Compare four O30 backends on identical HUG candidates and one metric mesh."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from anydexretarget.o30_retarget_backend import BACKENDS

ROOT = Path(__file__).resolve().parents[1]
RERANK = ROOT / "tools" / "rerank_o30_object_relative_candidates.py"
DEFAULT_BACKENDS = ("vector", "adaptive", "dexpilot", "joint_angle")


def _backends(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or set(result) - set(BACKENDS):
        raise argparse.ArgumentTypeError("backends must be a non-empty subset of " + ",".join(BACKENDS))
    return result


def _run(command: list[str]) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
    if process.wait():
        raise RuntimeError("".join(tail)[-3000:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-profile", type=Path)
    parser.add_argument("--backends", type=_backends, default=list(DEFAULT_BACKENDS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    parser.add_argument("--max-evaluations", type=int, default=180)
    parser.add_argument("--collision-max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--collision-max-iterations", type=int, default=160)
    parser.add_argument("--wrist-translation-limit-mm", type=float, default=30.0)
    parser.add_argument("--wrist-rotation-limit-deg", type=float, default=20.0)
    parser.add_argument("--closure-steps", type=int, default=32)
    parser.add_argument("--strict-top-k", type=int, default=8, help="Forwarded to the reranker; 0 validates every candidate.")
    args = parser.parse_args()
    if not args.candidates_dir.is_dir() or not args.object_mesh.is_file():
        raise FileNotFoundError("--candidates-dir or --object-mesh does not exist")
    if args.scale_profile is not None and not args.scale_profile.is_file():
        raise FileNotFoundError(args.scale_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for backend in args.backends:
        directory = args.output_dir / backend
        command = [str(RERANK), "--candidates-dir", str(args.candidates_dir), "--object-mesh", str(args.object_mesh), "--output-dir", str(directory), "--backend", backend, "--near-surface-gap-mm", str(args.near_surface_gap_mm), "--max-evaluations", str(args.max_evaluations), "--collision-max-joint-delta-rad", str(args.collision_max_joint_delta_rad), "--collision-max-iterations", str(args.collision_max_iterations), "--wrist-translation-limit-mm", str(args.wrist_translation_limit_mm), "--wrist-rotation-limit-deg", str(args.wrist_rotation_limit_deg), "--closure-steps", str(args.closure_steps), "--strict-top-k", str(args.strict_top_k)]
        if args.scale_profile is not None:
            command.extend(["--scale-profile", str(args.scale_profile)])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        row: dict[str, object] = {"backend": backend, "status": "failed"}
        try:
            _run(command)
            summary = json.loads((directory / "best_o30_candidates.json").read_text(encoding="utf-8"))
            best = (summary.get("recommended_top_candidates") or summary.get("top_candidates") or [None])[0]
            row.update({"status": "success", "successful_plan_count": summary["successful_plan_count"], "recommended_plan_count": summary["recommended_plan_count"], "best_candidate": "" if best is None else best["candidate"], "best_recommended": False if best is None else bool(best["recommended"]), "best_score": "" if best is None else float(best["final_o30_score"]), "summary": str((directory / "best_o30_candidates.json").resolve())})
        except Exception as exc:
            row["failure"] = str(exc)
        rows.append(row)
    successful = [row for row in rows if row["status"] == "success" and row.get("best_score", "") != ""]
    successful.sort(key=lambda row: (not bool(row["best_recommended"]), float(row["best_score"])))
    for rank, row in enumerate(successful, start=1):
        row["backend_rank"] = rank
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "backend_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"schema": "anydexretarget.o30_backend_benchmark/v1", "simulation_only": True, "hardware_command_generated": False, "method": "same HUG candidates plus same metric mesh plus O30-specific contact and strict full-mesh ranking", "backends": args.backends, "rows": rows, "csv": str((args.output_dir / "backend_benchmark.csv").resolve())}
    (args.output_dir / "backend_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("O30 backend benchmark complete: {}".format(args.output_dir / "backend_benchmark.json"), flush=True)


if __name__ == "__main__":
    main()
