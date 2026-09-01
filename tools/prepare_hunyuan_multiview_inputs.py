#!/usr/bin/env python3
"""Prepare masked RGB views for the official Hunyuan3D-2mv shape pipeline.

The model expects front/left/back images of the same object.  This tool makes
RGBA images from existing RGB-D capture directories without modifying them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--views", type=Path, nargs=3, required=True, metavar=("FRONT", "LEFT", "BACK"),
        help="Three capture directories, each containing rgb.png and object_mask.png.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=16, help="Extra pixels retained around each mask.")
    return parser.parse_args()


def read_view(directory: Path, padding: int) -> tuple[np.ndarray, dict[str, int | str]]:
    rgb_path = directory / "rgb.png"
    mask_path = directory / "object_mask.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(f"Unable to read {rgb_path}")
    if mask is None:
        raise FileNotFoundError(f"Unable to read {mask_path}")
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"RGB/mask dimensions differ in {directory}")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError(f"Mask is empty: {mask_path}")
    height, width = mask.shape
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    rgb = rgb[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba, {"source": str(directory.resolve()), "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def main() -> None:
    args = parse_args()
    if args.padding < 0:
        raise ValueError("--padding must be non-negative")
    labels = ("front", "left", "back")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "format": "hunyuan3d-2mv-rgba-v1",
        "view_order": list(labels),
        "note": "View labels are supplied by the caller; verify physical turntable order before inference.",
        "padding_px": args.padding,
        "views": {},
    }
    for label, directory in zip(labels, args.views):
        rgba, info = read_view(directory, args.padding)
        path = args.output / f"{label}.png"
        if not cv2.imwrite(str(path), rgba):
            raise RuntimeError(f"Unable to write {path}")
        metadata["views"][label] = {**info, "output": str(path.resolve())}
        print(f"{label}: {path} ({rgba.shape[1]}x{rgba.shape[0]})")
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
