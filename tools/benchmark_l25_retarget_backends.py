#!/usr/bin/env python3
"""Compare L25 retarget backends on identical HUG candidates and object mesh.

Each backend is passed through the same object-anchor planning, conservative
MuJoCo collision refinement and final feasibility gates.  This is offline
only; it never accesses hardware or CAN.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from anydexretarget.l25_retarget_backend import BACKENDS, ADAPTIVE_CONFIG


ROOT = Path(__file__).resolve().parents[1]
RERANK_TOOL = ROOT / "tools/rerank_l25_object_relative_candidates.py"
MESH_PROXY_TOOL = ROOT / "tools/prepare_mujoco_mesh_proxy.py"
DEFAULT_BACKENDS = ("vector", "adaptive", "dexpilot", "joint_angle")


def _parse_backends(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(BACKENDS))
    if unknown:
        raise argparse.ArgumentTypeError("Unknown backend(s): " + ", ".join(unknown))
    if not result:
        raise argparse.ArgumentTypeError("At least one backend is required")
    return result


def _run(command: list[str]) -> None:
    process = subprocess.Popen(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1,
    )
    output_tail: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_tail.append(line)
            if len(output_tail) > 80:
                output_tail.pop(0)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if return_code:
        raise RuntimeError("".join(output_tail)[-3000:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backends", type=_parse_backends, default=list(DEFAULT_BACKENDS),
                        help="Comma-separated backend list; default: vector,adaptive,dexpilot,joint_angle.")
    parser.add_argument("--adaptive-config", type=Path,
                        default=ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25_geometry_calibrated.yaml")
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    parser.add_argument("--max-evaluations", type=int, default=160)
    parser.add_argument("--collision-max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--max-penetration-mm", type=float, default=5.0)
    parser.add_argument("--max-self-penetration-mm", type=float, default=0.5)
    parser.add_argument("--max-contact-mean-error-mm", type=float, default=10.0)
    parser.add_argument(
        "--max-tip-mean-error-mm",
        type=float,
        dest="max_contact_mean_error_mm",
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--min-joint-margin", type=float, default=0.01)
    parser.add_argument("--max-mesh-faces", type=int, default=180_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.candidates_dir.is_dir() or not args.object_mesh.is_file():
        raise FileNotFoundError("--candidates-dir or --object-mesh does not exist")
    if "adaptive" in args.backends and not args.adaptive_config.is_file():
        raise FileNotFoundError(args.adaptive_config)
    if min(args.near_surface_gap_mm, args.max_evaluations, args.collision_max_joint_delta_rad) <= 0:
        parser.error("Invalid planning limits")
    if (
        args.max_penetration_mm < 0
        or args.max_self_penetration_mm < 0
        or args.max_contact_mean_error_mm < 0
        or not 0 <= args.min_joint_margin <= 0.5
        or not 0 < args.max_mesh_faces < 200_000
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("Acceptance gates must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = args.output_dir / "_shared"
    mesh_proxy = shared_dir / "object_collision_proxy.ply"
    contact_cache = shared_dir / "contact_plans"
    print("\n=== shared geometry cache ===")
    _run([
        sys.executable,
        "-u",
        str(MESH_PROXY_TOOL),
        "--mesh",
        str(args.object_mesh),
        "--output",
        str(mesh_proxy),
        "--max-faces",
        str(args.max_mesh_faces),
    ])
    rows: list[dict[str, object]] = []
    for backend in args.backends:
        backend_dir = args.output_dir / backend
        command = [
            sys.executable, "-u", str(RERANK_TOOL),
            "--candidates-dir", str(args.candidates_dir),
            "--object-mesh", str(args.object_mesh),
            "--output-dir", str(backend_dir),
            "--mesh-proxy", str(mesh_proxy),
            "--contact-plans-cache-dir", str(contact_cache),
            "--backend", backend,
            "--near-surface-gap-mm", str(args.near_surface_gap_mm),
            "--max-evaluations", str(args.max_evaluations),
            "--collision-aware",
            "--collision-max-joint-delta-rad", str(args.collision_max_joint_delta_rad),
            "--max-penetration-mm", str(args.max_penetration_mm),
            "--max-self-penetration-mm", str(args.max_self_penetration_mm),
            "--max-contact-mean-error-mm", str(args.max_contact_mean_error_mm),
            "--min-joint-margin", str(args.min_joint_margin),
        ]
        if backend == "adaptive":
            command.extend(["--config", str(args.adaptive_config)])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print(f"\n=== {backend} ===")
        row: dict[str, object] = {"backend": backend, "status": "failed"}
        try:
            _run(command)
            summary_path = backend_dir / "best_l25_candidates.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            preferred = (summary.get("recommended_top_candidates") or summary.get("top_candidates") or [None])[0]
            row.update({
                "status": "success",
                "recommended_plan_count": int(summary["recommended_plan_count"]),
                "successful_plan_count": int(summary["successful_plan_count"]),
                "best_candidate": "" if preferred is None else preferred["candidate"],
                "best_recommended": False if preferred is None else bool(preferred["recommended"]),
                "best_contact_mean_error_mm": "" if preferred is None else float(preferred["contact_mean_error_mm"]),
                "best_tip_mean_error_mm": "" if preferred is None else float(preferred["contact_mean_error_mm"]),
                "best_max_penetration_mm": "" if preferred is None else float(preferred["mujoco_max_penetration_mm"]),
                "best_max_self_penetration_mm": "" if preferred is None else float(preferred["mujoco_max_self_penetration_mm"]),
                "best_score": "" if preferred is None else float(preferred["final_l25_score"]),
                "summary": str(summary_path.resolve()),
            })
        except Exception as exc:
            row["failure"] = str(exc)
        rows.append(row)

    successful = [row for row in rows if row["status"] == "success"]
    successful.sort(key=lambda row: (
        not bool(row["best_recommended"]),
        float(row["best_score"]),
    ))
    for rank, row in enumerate(successful, start=1):
        row["backend_rank"] = rank
    csv_path = args.output_dir / "backend_benchmark.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "method": "same HUG candidates + same object mesh + per-backend L25 plan + collision-aware final ranking",
        "backends": args.backends,
        "adaptive_config": str(args.adaptive_config.resolve()) if "adaptive" in args.backends else "",
        "shared_mesh_proxy": str(mesh_proxy.resolve()),
        "shared_contact_plans": str(contact_cache.resolve()),
        "rows": rows,
        "csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "backend_benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nL25 backend benchmark complete (simulation only)")
    print(f"  CSV: {csv_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
