#!/usr/bin/env python3
"""Check target-machine repositories and model assets against a deployment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path).expanduser().resolve()


def _tree_identity(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    items = [path] if path.is_file() else sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    for item in items:
        relative = Path(item.name) if path.is_file() else item.relative_to(path)
        if relative.parts and relative.parts[0] == ".locks":
            continue
        digest.update(str(relative).encode())
        with item.resolve().open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        count += 1
    return count, size, digest.hexdigest()


def _default_path(entry: dict, workspace: Path) -> Path:
    relative = entry.get("workspace_relative_path")
    return workspace / relative if relative else Path(entry["source_path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", action="append", type=_named_path, default=[])
    parser.add_argument("--asset", action="append", type=_named_path, default=[])
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workspace = args.workspace.expanduser().resolve()
    repo_overrides = dict(args.repo)
    asset_overrides = dict(args.asset)
    failures = []

    for name, expected in manifest["repositories"].items():
        path = repo_overrides.get(name, _default_path(expected, workspace))
        if not (path / ".git").exists():
            failures.append(f"{name}: missing repository at {path}")
            continue
        commit = subprocess.check_output(
            ("git", "-C", str(path), "rev-parse", "HEAD"), text=True
        ).strip()
        status = subprocess.check_output(
            ("git", "-C", str(path), "status", "--porcelain"), text=True
        ).strip()
        if commit != expected["commit"]:
            failures.append(f"{name}: commit {commit} != {expected['commit']}")
        if args.require_clean and status:
            failures.append(f"{name}: worktree is dirty")
        print(f"repo {name}: commit={'ok' if commit == expected['commit'] else 'MISMATCH'}")

    for name, expected in manifest["assets"].items():
        path = asset_overrides.get(name, _default_path(expected, workspace))
        if not path.exists():
            failures.append(f"{name}: missing asset at {path}")
            continue
        if args.verify_hashes:
            count, size, digest = _tree_identity(path)
            matches = (
                count == expected["file_count"]
                and size == expected["size_bytes"]
                and digest == expected["sha256_tree"]
            )
            if not matches:
                failures.append(f"{name}: content hash or size mismatch")
            print(f"asset {name}: {'ok' if matches else 'MISMATCH'}")
        else:
            print(f"asset {name}: present (hash skipped)")

    if failures:
        print("Deployment preflight failed:")
        for failure in failures:
            print("  " + failure)
        raise SystemExit(1)
    print("Deployment manifest preflight passed")


if __name__ == "__main__":
    main()
