#!/usr/bin/env python3
"""Re-rank HUG candidates after final L25 contact and collision planning."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from anydexretarget.l25_retarget_backend import BACKENDS


ROOT = Path(__file__).resolve().parents[1]
L25_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
CONTACT_TOOL = ROOT / "tools/extract_object_relative_contacts.py"
PLAN_TOOL = ROOT / "tools/plan_l25_rigid_object_relative_grasp.py"
SCENE_TOOL = ROOT / "tools/build_l25_object_relative_scene.py"
COLLISION_TOOL = ROOT / "tools/refine_l25_collision_aware.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh-proxy",
        type=Path,
        help="Shared pre-simplified source-frame object collision mesh.",
    )
    parser.add_argument(
        "--contact-plans-cache-dir",
        type=Path,
        help="Shared backend-independent HUG contact-plan cache.",
    )
    parser.add_argument("--near-surface-gap-mm", type=float, default=25.0)
    parser.add_argument("--max-evaluations", type=int, default=160)
    parser.add_argument("--backend", choices=BACKENDS, default="vector")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dex-scaling", type=float)
    parser.add_argument("--dex-project-dist", type=float)
    parser.add_argument("--dex-escape-dist", type=float)
    parser.add_argument("--required-opposed-fingers", type=int, default=1)
    parser.add_argument("--min-opposed-anchor-span-ratio", type=float, default=0.08)
    parser.add_argument("--max-penetration-mm", type=float, default=5.0)
    parser.add_argument("--max-self-penetration-mm", type=float, default=0.5)
    parser.add_argument("--max-contact-mean-error-mm", type=float, default=10.0)
    parser.add_argument("--max-tip-mean-error-mm", type=float, dest="max_contact_mean_error_mm",
                        help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument("--max-posture-delta-rms", type=float, default=0.35)
    parser.add_argument("--min-joint-margin", type=float, default=0.01)
    parser.add_argument("--hardware-reports-dir", type=Path)
    parser.add_argument("--max-hardware-mean-error", type=float, default=8.0)
    parser.add_argument("--require-hardware-validation", action="store_true")
    parser.add_argument("--build-scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-aware", action="store_true")
    parser.add_argument("--collision-max-joint-delta-rad", type=float, default=0.10)
    parser.add_argument("--limit", type=int, help="Process only the first N successful HUG candidates.")
    args = parser.parse_args()
    invalid = (
        args.near_surface_gap_mm <= 0
        or args.max_evaluations <= 0
        or args.required_opposed_fingers < 1
        or not 0 <= args.min_opposed_anchor_span_ratio <= 1
        or args.max_penetration_mm < 0
        or args.max_self_penetration_mm < 0
        or args.max_contact_mean_error_mm < 0
        or args.max_posture_delta_rms < 0
        or not 0 <= args.min_joint_margin <= 0.5
        or args.max_hardware_mean_error < 0
        or args.collision_max_joint_delta_rad <= 0
        or (args.limit is not None and args.limit <= 0)
    )
    if invalid:
        parser.error("Invalid re-ranking thresholds")
    if args.config is not None and args.backend in {"dexpilot", "joint_angle"}:
        parser.error("--config is only supported for native backends")
    if any(
        value is not None and value <= 0
        for value in (args.dex_scaling, args.dex_project_dist, args.dex_escape_dist)
    ):
        parser.error("Dex backend overrides must be positive")
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


def _contact_cache_valid(
    path: Path, canonical: Path, object_mesh: Path, near_surface_gap_mm: float
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            source_canonical = Path(
                str(data["source_canonical_grasp"].item())
            ).resolve()
            source_mesh = Path(str(data["source_object_mesh"].item())).resolve()
            gap_mm = float(np.asarray(data["near_surface_gap_m"]).item()) * 1000.0
    except (KeyError, OSError, ValueError):
        return False
    return (
        source_canonical == canonical.resolve()
        and source_mesh == object_mesh.resolve()
        and np.isclose(gap_mm, near_surface_gap_mm, atol=1e-6)
    )


def _scene_command(plan: Path, output_dir: Path, mesh_proxy: Path | None) -> list[str]:
    command = [
        sys.executable, str(SCENE_TOOL),
        "--plan", str(plan), "--output-dir", str(output_dir),
    ]
    if mesh_proxy is not None:
        command.extend(["--mesh-proxy", str(mesh_proxy)])
    return command


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(message[-3000:])


def _mesh_diagonal(path: Path) -> float:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [value for value in raw.geometry.values() if isinstance(value, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.vertices) == 0:
        raise ValueError("Object mesh has no vertices")
    diagonal = float(np.linalg.norm(np.asarray(raw.bounds[1] - raw.bounds[0], dtype=np.float64)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Object mesh has invalid metric bounds")
    return diagonal


def _joint_metrics(plan_path: Path) -> dict[str, float | int]:
    with np.load(plan_path, allow_pickle=False) as data:
        q = np.asarray(data["qpos_vector_order"], dtype=np.float64)
        names = [str(value) for value in data["vector_joint_names"]]
        q_initial = np.asarray(
            data["qpos_initial_retarget"] if "qpos_initial_retarget" in data.files else q,
            dtype=np.float64,
        )
    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    model_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    indices = {
        str(name).lower(): index
        for index, name in enumerate(model_names)
        if name is not None
    }
    joint_indices = np.asarray([indices[name.lower()] for name in names], dtype=np.int64)
    lower = model.jnt_range[joint_indices, 0]
    upper = model.jnt_range[joint_indices, 1]
    ranges = np.maximum(upper - lower, 1e-6)
    margins = np.minimum((q - lower) / ranges, (upper - q) / ranges)
    posture = (q - q_initial) / ranges
    return {
        "joint_saturation_count": int(np.count_nonzero(margins <= 0.05)),
        "joint_min_normalized_margin": float(margins.min()),
        "posture_delta_rms": float(np.sqrt(np.mean(posture**2))),
        "posture_delta_max_rad": float(np.max(np.abs(q - q_initial), initial=0.0)),
    }


def _contact_metrics(contact_path: Path, object_diagonal: float) -> dict[str, object]:
    with np.load(contact_path, allow_pickle=False) as data:
        active = np.asarray(data["near_surface"], dtype=np.uint8).astype(bool)
        normals = np.asarray(data["surface_normal_camera"], dtype=np.float64)
        anchors = np.asarray(data["surface_anchor_camera"], dtype=np.float64)
        names = [str(value).lower() for value in data["finger_names"]]
        kinds = [
            str(value)
            for value in data["contact_point_kind"]
        ] if "contact_point_kind" in data.files else ["tip"] * len(names)
    thumb = names.index("thumb")
    others = [index for index, enabled in enumerate(active) if enabled and index != thumb]
    dots = {
        names[index]: float(np.dot(normals[thumb], normals[index]))
        for index in others
    } if active[thumb] else {}
    opposed = [name for name, dot in dots.items() if dot <= -0.2]
    spans = {
        names[index]: float(np.linalg.norm(anchors[thumb] - anchors[index]) / object_diagonal)
        for index in others
    } if active[thumb] else {}
    opposed_spans = [spans[name] for name in opposed]
    return {
        "active_contact_fingers": int(active.sum()),
        "contact_point_kinds": ",".join(kinds),
        "thumb_opposed": bool(opposed),
        "thumb_opposed_fingers": ",".join(opposed),
        "thumb_opposed_finger_count": len(opposed),
        "thumb_best_normal_dot": min(dots.values(), default=1.0),
        "opposed_anchor_span_ratio": max(opposed_spans, default=0.0),
    }


def _scene_metrics(report_path: Path) -> dict[str, float | int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    object_contacts = report.get("contacts", [])
    self_contacts = report.get("self_contacts", [])
    object_penetration = max(
        (max(0.0, -float(item["distance_m"])) * 1000.0 for item in object_contacts),
        default=0.0,
    )
    self_penetration = max(
        (max(0.0, -float(item["distance_m"])) * 1000.0 for item in self_contacts),
        default=0.0,
    )
    return {
        "mujoco_contact_pairs": len(object_contacts),
        "mujoco_max_penetration_mm": object_penetration,
        "mujoco_cross_finger_contact_pairs": len(self_contacts),
        "mujoco_max_self_penetration_mm": self_penetration,
    }


def _hardware_metrics(directory: Path | None, candidate: str) -> dict[str, object]:
    result: dict[str, object] = {
        "hardware_reproduction_validated": False,
        "hardware_tracking_mean_error_0_255": None,
        "hardware_tracking_max_error_0_255": None,
        "hardware_report": "",
    }
    if directory is None:
        return result
    paths = (
        directory / f"{candidate}.json",
        directory / candidate / "hardware_report.json",
    )
    report_path = next((path for path in paths if path.is_file()), None)
    if report_path is None:
        return result
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = np.asarray(report.get("target_command_0_255", []), dtype=np.float64)
    measured = np.asarray(report.get("state_after_0_255", []), dtype=np.float64)
    if target.shape != (25,) or measured.shape != (25,):
        return result
    error = np.abs(target - measured)
    result.update({
        "hardware_reproduction_validated": True,
        "hardware_tracking_mean_error_0_255": float(error.mean()),
        "hardware_tracking_max_error_0_255": float(error.max()),
        "hardware_report": str(report_path.resolve()),
    })
    return result


def _score(row: dict[str, object], minimum_span: float) -> float:
    best_dot = float(row["thumb_best_normal_dot"])
    span = float(row["opposed_anchor_span_ratio"])
    opposition_penalty = 25.0 if not row["thumb_opposed"] else max(0.0, best_dot + 0.8) * 8.0
    span_penalty = max(0.0, minimum_span - span) * 100.0
    margin_penalty = max(
        0.0, 0.10 - float(row["joint_min_normalized_margin"])
    ) * 20.0
    hardware_error = row.get("hardware_tracking_mean_error_0_255")
    hardware_penalty = (
        0.0 if hardware_error is None else 0.5 * float(hardware_error)
    )
    return (
        float(row["contact_mean_error_mm"])
        + 0.5 * float(row["contact_max_error_mm"])
        + 3.0 * int(row["joint_saturation_count"])
        + 12.0 * float(row["mujoco_max_penetration_mm"])
        + 20.0 * float(row["mujoco_max_self_penetration_mm"])
        + 12.0 * float(row["posture_delta_rms"])
        + margin_penalty
        + hardware_penalty
        + opposition_penalty
        + span_penalty
    )


def main() -> None:
    args = _parse_args()
    if not args.candidates_dir.is_dir() or not args.object_mesh.is_file():
        raise FileNotFoundError("--candidates-dir or --object-mesh does not exist")
    if args.mesh_proxy is not None and not args.mesh_proxy.is_file():
        raise FileNotFoundError(args.mesh_proxy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_diagonal = _mesh_diagonal(args.object_mesh)
    rows: list[dict[str, object]] = []

    sources = _candidates(args.candidates_dir)
    if args.limit is not None:
        sources = sources[:args.limit]
    for source in sources:
        candidate = source["candidate"]
        candidate_dir = args.output_dir / candidate
        contact_path = (
            args.contact_plans_cache_dir / candidate / "contact_plan.npz"
            if args.contact_plans_cache_dir is not None
            else candidate_dir / "contact_plan.npz"
        )
        plan_path = candidate_dir / "l25_object_relative_plan.npz"
        scene_dir = candidate_dir / "scene"
        row: dict[str, object] = {
            "candidate": candidate,
            "seed": source.get("seed", ""),
            "status": "failed",
        }
        try:
            canonical_path = (
                args.candidates_dir / candidate / "canonical_grasp.npz"
            )
            contact_cache_hit = _contact_cache_valid(
                contact_path, canonical_path, args.object_mesh,
                args.near_surface_gap_mm,
            )
            print(
                f"{candidate}: contact cache "
                f"{'hit' if contact_cache_hit else 'miss'}",
                flush=True,
            )
            if not contact_cache_hit:
                _run([
                    sys.executable, str(CONTACT_TOOL),
                    "--canonical-grasp", str(canonical_path),
                    "--object-mesh", str(args.object_mesh),
                    "--output", str(contact_path),
                    "--near-surface-gap-mm", str(args.near_surface_gap_mm),
                ])
            plan_command = [
                sys.executable, str(PLAN_TOOL),
                "--contact-plan", str(contact_path),
                "--output", str(plan_path),
                "--max-evaluations", str(args.max_evaluations),
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

            if args.build_scenes or args.collision_aware:
                _run(_scene_command(plan_path, scene_dir, args.mesh_proxy))
            if args.collision_aware:
                collision_plan = candidate_dir / "l25_collision_aware_plan.npz"
                collision_scene_dir = candidate_dir / "collision_aware_scene"
                _run([
                    sys.executable, str(COLLISION_TOOL),
                    "--plan", str(plan_path),
                    "--scene-xml", str(scene_dir / "l25_object_relative_scene.xml"),
                    "--output", str(collision_plan),
                    "--max-joint-delta-rad", str(args.collision_max_joint_delta_rad),
                ])
                plan_path = collision_plan
                scene_dir = collision_scene_dir
                _run(_scene_command(plan_path, scene_dir, args.mesh_proxy))

            report = json.loads(plan_path.with_suffix(".json").read_text(encoding="utf-8"))
            errors_mm = [
                float(value)
                for value in report.get(
                    "active_contact_error_after_mm",
                    report["active_fingertip_error_after_mm"],
                )
            ]
            row.update(_contact_metrics(contact_path, object_diagonal))
            row.update(_joint_metrics(plan_path))
            if args.build_scenes or args.collision_aware:
                row.update(_scene_metrics(scene_dir / "scene_report.json"))
            else:
                row.update({
                    "mujoco_contact_pairs": 0,
                    "mujoco_max_penetration_mm": 0.0,
                    "mujoco_cross_finger_contact_pairs": 0,
                    "mujoco_max_self_penetration_mm": 0.0,
                })
            row.update({
                "status": "success",
                "contact_mean_error_mm": float(np.mean(errors_mm)),
                "contact_max_error_mm": float(np.max(errors_mm)),
                "tip_mean_error_mm": float(np.mean(errors_mm)),
                "tip_max_error_mm": float(np.max(errors_mm)),
                "collision_aware": bool(args.collision_aware),
                "plan": str(plan_path.resolve()),
                "scene": str(scene_dir.resolve()) if args.build_scenes or args.collision_aware else "",
            })
            row.update(_hardware_metrics(args.hardware_reports_dir, candidate))
            row["final_l25_score"] = _score(row, args.min_opposed_anchor_span_ratio)
            row["topology_valid"] = bool(
                int(row["thumb_opposed_finger_count"]) >= args.required_opposed_fingers
                and float(row["opposed_anchor_span_ratio"]) >= args.min_opposed_anchor_span_ratio
            )
            row["collision_valid"] = bool(
                float(row["mujoco_max_penetration_mm"]) <= args.max_penetration_mm
                and float(row["mujoco_max_self_penetration_mm"]) <= args.max_self_penetration_mm
            )
            row["contact_targets_valid"] = bool(
                float(row["contact_mean_error_mm"]) <= args.max_contact_mean_error_mm
            )
            row["posture_valid"] = bool(
                float(row["posture_delta_rms"]) <= args.max_posture_delta_rms
            )
            row["joint_margin_valid"] = bool(
                float(row["joint_min_normalized_margin"]) >= args.min_joint_margin
            )
            row["hardware_valid"] = bool(
                (
                    not args.require_hardware_validation
                    and not row["hardware_reproduction_validated"]
                )
                or (
                    row["hardware_reproduction_validated"]
                    and float(row["hardware_tracking_mean_error_0_255"])
                    <= args.max_hardware_mean_error
                )
            )
            row["recommended"] = bool(
                row["topology_valid"]
                and row["collision_valid"]
                and row["contact_targets_valid"]
                and row["posture_valid"]
                and row["joint_margin_valid"]
                and row["hardware_valid"]
            )
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
        "score_version": "l25_contact_topology_v6_hardware_optional",
        "interpretation": (
            "Lower is better. Contact count is diagnostic only and is not a score term. "
            "Ranking uses distal-pad target error, thumb opposition and anchor span, "
            "object penetration, cross-finger self-penetration, joint margin, and "
            "drift from the initial retargeted pose. Hardware command-tracking "
            "error is included when a per-candidate report is supplied; this does "
            "not establish force or grasp stability."
        ),
        "candidate_count": len(rows),
        "retarget_backend": args.backend,
        "object_diagonal_m": object_diagonal,
        "mesh_proxy": (
            "" if args.mesh_proxy is None else str(args.mesh_proxy.resolve())
        ),
        "contact_plans_cache_dir": (
            ""
            if args.contact_plans_cache_dir is None
            else str(args.contact_plans_cache_dir.resolve())
        ),
        "successful_plan_count": len(successful),
        "recommended_plan_count": len(recommended),
        "gates": {
            "required_opposed_fingers": args.required_opposed_fingers,
            "min_opposed_anchor_span_ratio": args.min_opposed_anchor_span_ratio,
            "max_penetration_mm": args.max_penetration_mm,
            "max_self_penetration_mm": args.max_self_penetration_mm,
            "max_contact_mean_error_mm": args.max_contact_mean_error_mm,
            "max_posture_delta_rms": args.max_posture_delta_rms,
            "min_joint_margin": args.min_joint_margin,
            "require_hardware_validation": args.require_hardware_validation,
            "max_hardware_mean_error_0_255": args.max_hardware_mean_error,
        },
        "top_candidates": successful[:3],
        "recommended_top_candidates": recommended[:3],
        "csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "best_l25_candidates.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"L25 final-plan v6 ranking complete: {len(successful)}/{len(rows)} planned")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
