#!/usr/bin/env python3
"""Record reproducible repository and model-asset identities for robot deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path).expanduser().resolve()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(path), *args), text=True
    ).strip()


def _repo(path: Path, workspace: Path) -> dict:
    status = _git(path, "status", "--porcelain")
    try:
        relative = str(path.relative_to(workspace))
    except ValueError:
        relative = None
    return {
        "source_path": str(path),
        "workspace_relative_path": relative,
        "commit": _git(path, "rev-parse", "HEAD"),
        "branch": _git(path, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "remotes": _git(path, "remote", "-v").splitlines(),
    }


def _hash_file(path: Path, digest: hashlib._Hash) -> int:
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size


def _asset(path: Path, workspace: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    file_count = 0
    size = 0
    if path.is_file():
        digest.update(path.name.encode())
        size = _hash_file(path, digest)
        file_count = 1
    else:
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = item.relative_to(path)
            if relative.parts and relative.parts[0] == ".locks":
                continue
            digest.update(str(relative).encode())
            size += _hash_file(item.resolve(), digest)
            file_count += 1
    try:
        relative = str(path.relative_to(workspace))
    except ValueError:
        relative = None
    return {
        "source_path": str(path),
        "workspace_relative_path": relative,
        "kind": "file" if path.is_file() else "directory",
        "file_count": file_count,
        "size_bytes": size,
        "sha256_tree": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", action="append", type=_named_path, default=[])
    parser.add_argument("--asset", action="append", type=_named_path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_workspace": str(workspace),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "repositories": {
            name: _repo(path, workspace) for name, path in args.repo
        },
        "assets": {
            name: _asset(path, workspace) for name, path in args.asset
        },
        "deployment_policy": {
            "copy_virtualenv": False,
            "copy_ros_install": False,
            "rebuild_on_target": True,
            "require_clean_repository_commits": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dirty = [
        name for name, repo in manifest["repositories"].items() if repo["dirty"]
    ]
    print(f"Deployment manifest written: {args.output}")
    print(f"  repositories: {len(manifest['repositories'])}")
    print(f"  assets: {len(manifest['assets'])}")
    if dirty:
        print("  WARNING dirty repositories: " + ", ".join(dirty))


if __name__ == "__main__":
    main()
