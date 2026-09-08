#!/usr/bin/env python3
"""Re-rank HUG O30 Vector candidates with mesh self/object collision checks.

This command is offline-only. It writes collision reports and a conservative
safe-close trajectory for each candidate; it never publishes robot commands.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from anydexretarget.luban_contract import o30_qpos_to_luban_active
from anydexretarget.o30_collision import O30CollisionEvaluator


def _load_records(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No trajectory records: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Invalid trajectory records: {path}")
    return records


def _load_qpos(path: Path) -> np.ndarray:
    records = _load_records(path)
    qpos = np.asarray(records[-1]["target"], dtype=np.float64)
    if qpos.shape != (20,) or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid O30 qpos in {path}")
    return qpos


def _write_safe_close_trajectory(
    source: Path,
    output: Path,
    safe_qpos: np.ndarray,
    *,
    safe_fraction: float,
    frames: int,
) -> None:
    """Write an O30 open-to-safe-close command trajectory for offline playback."""
    records = _load_records(source)
    target = np.asarray(safe_qpos, dtype=np.float32)
    if target.shape != (20,):
        raise ValueError("safe O30 qpos must contain 20 values")
    template = records[-1]
    timestamps = np.linspace(0.0, (frames - 1) / 30.0, frames)
    safe_records: list[dict] = []
    for fraction, timestamp in zip(np.linspace(0.0, 1.0, frames), timestamps):
        qpos = target * fraction
        record = copy.deepcopy(template)
        record["timestamp"] = float(timestamp)
        record["target"] = qpos.copy()
        record["sim_qpos"] = qpos.copy()
        record["luban_hand_positions"] = o30_qpos_to_luban_active(qpos).astype(np.float32)
        record["collision_filtered"] = True
        record["safe_close_fraction"] = float(safe_fraction)
        safe_records.append(record)
    with output.open("wb") as stream:
        pickle.dump(safe_records, stream)


def _load_object(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points_camera"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Invalid points_camera: {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--closure-steps", type=int, default=16)
    parser.add_argument("--forbidden-clearance-mm", type=float, default=3.0)
    parser.add_argument("--mesh-vertices-per-link", type=int, default=384)
    parser.add_argument("--safe-close-frames", type=int, default=16)
    args = parser.parse_args()
    if (
        args.closure_steps < 2
        or args.forbidden_clearance_mm <= 0
        or args.mesh_vertices_per_link < 32
        or args.safe_close_frames < 2
    ):
        parser.error("invalid collision sampling parameters")
    candidates = sorted(path for path in args.candidates_dir.glob("candidate_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No candidate directories: {args.candidates_dir}")
    object_points = _load_object(args.pointcloud)
    rows = []
    for directory in candidates:
        with np.load(directory / "canonical_grasp.npz", allow_pickle=False) as data:
            evaluator = O30CollisionEvaluator(
                object_points,
                wrist_position_camera=data["wrist_position_camera"],
                canonical_basis_row=data["canonical_basis_row"],
                mesh_vertices_per_link=args.mesh_vertices_per_link,
                forbidden_clearance_m=args.forbidden_clearance_mm / 1000.0,
            )
        qpos = _load_qpos(directory / "trajectory.pkl")
        target = evaluator.evaluate(qpos)
        safe_fraction, close = evaluator.safe_closure(qpos, steps=args.closure_steps)
        safe_qpos = qpos * safe_fraction
        safe_trajectory = directory / "safe_close_trajectory.pkl"
        _write_safe_close_trajectory(
            directory / "trajectory.pkl",
            safe_trajectory,
            safe_qpos,
            safe_fraction=safe_fraction,
            frames=args.safe_close_frames,
        )
        report = {
            "candidate": directory.name,
            "target": target.as_dict(),
            "safe_close_fraction": safe_fraction,
            "safe_close": close.as_dict(),
            "safe_close_qpos": safe_qpos.tolist(),
            "safe_close_trajectory": str(safe_trajectory),
            "closure_steps": args.closure_steps,
            "forbidden_clearance_mm": args.forbidden_clearance_mm,
            "mesh_vertices_per_link": args.mesh_vertices_per_link,
        }
        (directory / "o30_collision_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Lexicographic: feasible closing path first, then fewer collision violations,
        # then retain the existing HUG/Vector score as a final tie breaker.
        old = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        rank_key = (
            safe_fraction < 0.999999,
            target.forbidden_object_collision_count,
            target.self_collision_count,
            float(old.get("total_score", float("inf"))),
        )
        rows.append({
            "candidate": directory.name,
            "safe_close_fraction": safe_fraction,
            "safe_close_trajectory": str(safe_trajectory),
            "target_self_collision_count": target.self_collision_count,
            "target_forbidden_object_collision_count": target.forbidden_object_collision_count,
            "target_fingertip_contact_count": target.fingertip_contact_count,
            "safe_close_fingertip_contact_count": close.fingertip_contact_count,
            "hug_vector_score": float(old.get("total_score", float("inf"))),
            "rank_key": rank_key,
        })
        print(
            f"{directory.name}: safe_close={safe_fraction:.3f} "
            f"self={target.self_collision_count} "
            f"forbidden={target.forbidden_object_collision_count} "
            f"tips={target.fingertip_contact_count}"
        )
    rows.sort(key=lambda row: row.pop("rank_key"))
    for rank, row in enumerate(rows, start=1):
        row["collision_rank"] = rank
    summary = {
        "schema": "anydexretarget.o30_collision_ranking/v1",
        "pointcloud": str(args.pointcloud.resolve()),
        "candidate_count": len(rows),
        "selection": "safe linear closure, then target object/self collision counts, then prior HUG Vector score",
        "best_candidate": rows[0],
        "candidates": rows,
    }
    output = args.candidates_dir / "o30_collision_ranking.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"collision-aware O30 ranking: {output}")


if __name__ == "__main__":
    main()
