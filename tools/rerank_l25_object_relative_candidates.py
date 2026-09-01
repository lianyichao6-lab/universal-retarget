#!/usr/bin/env python3
"""Re-rank HUG candidates after L25 object-relative planning.

The first HUG ranking estimates MANO plausibility and initial L25 feasibility.
This offline-only tool applies the actual L25 mesh-contact refinement to every
candidate, then ranks those final robot plans using explicit geometric metrics.
It never connects to hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.l25_retarget_backend import BACKENDS


ROOT = Path(__file__).resolve().parents[1]
L25_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
CONTACT_TOOL = ROOT / "tools/extract_object_relative_contacts.py"
PLAN_TOOL = ROOT / "tools/plan_l25_rigid_object_relative_grasp.py"
SCENE_TOOL = ROOT / "tools/build_l25_object_relative_scene.py"
COLLISION_REFINE_TOOL = ROOT / "tools/refine_l25_collision_aware.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    parser.add_argument("--max-evaluations", type=int, default=160)
    parser.add_argument("--backend", choices=BACKENDS, default="vector")
    parser.add_argument("--config", type=Path, help="Optional native Vector/Adaptive YAML override.")
    parser.add_argument("--dex-scaling", type=float)
    parser.add_argument("--dex-project-dist", type=float)
    parser.add_argument("--dex-escape-dist", type=float)
    parser.add_argument("--required-opposed-fingers", type=int, default=1, help="Minimum non-thumb fingers on an opposing mesh surface for a recommended grasp.")
    parser.add_argument("--max-penetration-mm", type=float, default=5.0, help="Maximum MuJoCo mesh penetration allowed for a recommended grasp.")
    parser.add_argument("--max-tip-mean-error-mm", type=float, default=10.0,
                        help="Maximum mean active-fingertip target error allowed for a recommended grasp.")
    parser.add_argument("--build-scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-aware", action="store_true",
                        help="Conservatively refine each final plan against MuJoCo object-mesh penetration.")
    parser.add_argument("--collision-max-joint-delta-rad", type=float, default=0.10,
                        help="Maximum per-joint movement during collision-aware refinement.")
    args = parser.parse_args()
    if (args.near_surface_gap_mm <= 0 or args.max_evaluations <= 0
            or args.required_opposed_fingers < 1 or args.max_penetration_mm < 0
            or args.collision_max_joint_delta_rad <= 0
            or args.max_tip_mean_error_mm < 0):
        parser.error("Invalid re-ranking thresholds")
    if args.config is not None and args.backend in {"dexpilot", "joint_angle"}:
        parser.error("--config is only supported for native Vector/Adaptive backends")
    if any(value is not None and value <= 0 for value in
           (args.dex_scaling, args.dex_project_dist, args.dex_escape_dist)):
        parser.error("Dex backend overrides must be positive")
    if (args.dex_project_dist is not None and args.dex_escape_dist is not None
            and args.dex_escape_dist < args.dex_project_dist):
        parser.error("--dex-escape-dist must be >= --dex-project-dist")
    return args


def _candidates(path: Path) -> list[dict[str, str]]:
    ranking = path / "candidates.csv"
    if not ranking.is_file():
        raise FileNotFoundError(ranking)
    with ranking.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = [row for row in rows if row.get("status") == "success"]
    if not result:
        raise ValueError("No successful HUG candidates to re-rank")
    return result


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(message[-2000:])


def _joint_metrics(plan_path: Path) -> tuple[int, float]:
    with np.load(plan_path, allow_pickle=False) as data:
        q = np.asarray(data["qpos_vector_order"], dtype=np.float64)
        names = [str(value) for value in data["vector_joint_names"]]
    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    indices = {str(name).lower(): index for index, name in enumerate(model_names) if name is not None}
    joint_indices = np.asarray([indices[name.lower()] for name in names], dtype=np.int64)
    lower, upper = model.jnt_range[joint_indices, 0], model.jnt_range[joint_indices, 1]
    normalized_margin = np.minimum((q - lower) / (upper - lower), (upper - q) / (upper - lower))
    return int(np.count_nonzero(normalized_margin <= 0.05)), float(normalized_margin.min())


def _contact_metrics(contact_path: Path) -> tuple[int, list[str], float]:
    with np.load(contact_path, allow_pickle=False) as data:
        active = np.asarray(data["near_surface"], dtype=np.uint8).astype(bool)
        normals = np.asarray(data["surface_normal_camera"], dtype=np.float64)
        names = [str(value).lower() for value in data["finger_names"]]
    thumb = names.index("thumb")
    others = [index for index, enabled in enumerate(active) if enabled and index != thumb]
    if not active[thumb] or not others:
        return int(active.sum()), [], 1.0
    dots = {names[index]: float(np.dot(normals[thumb], normals[index])) for index in others}
    opposed = [name for name, dot in dots.items() if dot <= -0.2]
    return int(active.sum()), opposed, min(dots.values())


def _scene_metrics(report_path: Path) -> tuple[int, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    distances = [float(contact["distance_m"]) for contact in report.get("contacts", [])]
    penetration_mm = max((max(0.0, -distance) * 1000.0 for distance in distances), default=0.0)
    return int(report.get("contact_pair_count", 0)), penetration_mm


def _score(mean_error_mm: float, max_error_mm: float, saturation: int,
           thumb_opposed: bool, penetration_mm: float) -> float:
    """Transparent heuristic; lower means more L25-geometrically feasible."""
    return (
        mean_error_mm
        + 0.5 * max_error_mm
        + saturation * 5.0
        + (0.0 if thumb_opposed else 25.0)
        + penetration_mm * 10.0
    )


def main() -> None:
    args = _parse_args()
    if not args.candidates_dir.is_dir() or not args.object_mesh.is_file():
        raise FileNotFoundError("--candidates-dir or --object-mesh does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for source in _candidates(args.candidates_dir):
        candidate = source["candidate"]
        candidate_dir = args.output_dir / candidate
        contact_path = candidate_dir / "contact_plan.npz"
        plan_path = candidate_dir / "l25_object_relative_plan.npz"
        scene_dir = candidate_dir / "scene"
        row: dict[str, object] = {"candidate": candidate, "seed": source.get("seed", ""), "status": "failed"}
        try:
            _run([
                sys.executable, str(CONTACT_TOOL), "--canonical-grasp",
                str(args.candidates_dir / candidate / "canonical_grasp.npz"), "--object-mesh", str(args.object_mesh),
                "--output", str(contact_path), "--near-surface-gap-mm", str(args.near_surface_gap_mm),
            ])
            plan_command = [
                sys.executable, str(PLAN_TOOL), "--contact-plan", str(contact_path),
                "--output", str(plan_path), "--max-evaluations", str(args.max_evaluations),
                "--backend", args.backend,
            ]
            if args.config is not None:
                plan_command.extend(["--config", str(args.config)])
            if args.dex_scaling is not None:
                plan_command.extend(["--dex-scaling", str(args.dex_scaling)])
            if args.dex_project_dist is not None:
                plan_command.extend(["--dex-project-dist", str(args.dex_project_dist)])
            if args.dex_escape_dist is not None:
                plan_command.extend(["--dex-escape-dist", str(args.dex_escape_dist)])
            _run(plan_command)
            # Preserve the baseline contact-target plan, then optionally score a
            # separate conservative MuJoCo collision-aware refinement.
            if args.build_scenes or args.collision_aware:
                _run([
                    sys.executable, str(SCENE_TOOL), "--plan", str(plan_path), "--output-dir", str(scene_dir),
                ])
            if args.collision_aware:
                collision_plan = candidate_dir / "l25_collision_aware_plan.npz"
                collision_scene_dir = candidate_dir / "collision_aware_scene"
                _run([
                    sys.executable, str(COLLISION_REFINE_TOOL), "--plan", str(plan_path),
                    "--scene-xml", str(scene_dir / "l25_object_relative_scene.xml"),
                    "--output", str(collision_plan), "--max-joint-delta-rad",
                    str(args.collision_max_joint_delta_rad),
                ])
                plan_path = collision_plan
                scene_dir = collision_scene_dir
                _run([
                    sys.executable, str(SCENE_TOOL), "--plan", str(plan_path), "--output-dir", str(scene_dir),
                ])
            plan_report = json.loads(plan_path.with_suffix(".json").read_text(encoding="utf-8"))
            active_count, opposed_fingers, thumb_dot = _contact_metrics(contact_path)
            thumb_opposed = bool(opposed_fingers)
            errors_mm = [float(value) for value in plan_report["active_fingertip_error_after_mm"]]
            saturation, min_margin = _joint_metrics(plan_path)
            contact_pairs, penetration_mm = 0, 0.0
            if args.build_scenes or args.collision_aware:
                contact_pairs, penetration_mm = _scene_metrics(scene_dir / "scene_report.json")
            row.update({
                "status": "success",
                "active_contact_fingers": active_count,
                "thumb_opposed": thumb_opposed,
                "thumb_opposed_fingers": ",".join(opposed_fingers),
                "thumb_opposed_finger_count": len(opposed_fingers),
                "thumb_best_normal_dot": thumb_dot,
                "tip_mean_error_mm": float(np.mean(errors_mm)),
                "tip_max_error_mm": float(np.max(errors_mm)),
                "joint_saturation_count": saturation,
                "joint_min_normalized_margin": min_margin,
                "mujoco_contact_pairs": contact_pairs,
                "mujoco_max_penetration_mm": penetration_mm,
                "collision_aware": bool(args.collision_aware),
                "plan": str(plan_path.resolve()),
                "scene": str(scene_dir.resolve()) if args.build_scenes or args.collision_aware else "",
            })
            row["final_l25_score"] = _score(
                row["tip_mean_error_mm"], row["tip_max_error_mm"], saturation,
                thumb_opposed, penetration_mm,
            )
            row["topology_valid"] = len(opposed_fingers) >= args.required_opposed_fingers
            row["collision_valid"] = penetration_mm <= args.max_penetration_mm
            row["tip_targets_valid"] = row["tip_mean_error_mm"] <= args.max_tip_mean_error_mm
            row["recommended"] = bool(row["topology_valid"] and row["collision_valid"] and row["tip_targets_valid"])
        except Exception as exc:
            row["failure"] = str(exc)
        rows.append(row)
        print(f"{candidate}: {row['status']}")

    successful = [row for row in rows if row["status"] == "success"]
    successful.sort(key=lambda row: float(row["final_l25_score"]))
    for rank, row in enumerate(successful, start=1):
        row["final_l25_rank"] = rank
    recommended = [row for row in successful if row.get("recommended")]
    csv_path = args.output_dir / "l25_final_candidates.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "score_version": "l25_object_relative_geometric_v4_collision_aware",
        "interpretation": "Lower score is a transparent geometric heuristic, not force-closure validation. Active contact count is a planning precondition and diagnostic, not a score term. A candidate is recommended only when thumb opposition, mesh-penetration, and active-tip-error gates all pass. When --collision-aware is set, scoring uses a separate conservative MuJoCo-refined plan.",
        "candidate_count": len(rows),
        "retarget_backend": args.backend,
        "backend_config": "" if args.config is None else str(args.config.resolve()),
        "successful_plan_count": len(successful),
        "recommended_plan_count": len(recommended),
        "required_opposed_fingers": args.required_opposed_fingers,
        "max_penetration_mm": args.max_penetration_mm,
        "max_tip_mean_error_mm": args.max_tip_mean_error_mm,
        "top_candidates": successful[:3],
        "recommended_top_candidates": recommended[:3],
        "csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "best_l25_candidates.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"L25 final-plan re-ranking complete: {len(successful)}/{len(rows)} planned")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
