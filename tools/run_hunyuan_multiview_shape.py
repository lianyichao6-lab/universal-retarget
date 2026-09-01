#!/usr/bin/env python3
"""Generate an untextured mesh from three prepared Hunyuan3D-2mv views.

Run this file with the isolated ``.venv-hunyuan3d`` environment.  The output is
in Hunyuan's canonical frame; it is intentionally *not* yet in the RGB-D
camera frame and must be aligned before it can replace a HUG point cloud.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Directory containing front.png, left.png, back.png.")
    parser.add_argument("--output", type=Path, required=True, help="Output .glb or .ply mesh path.")
    parser.add_argument(
        "--model-dir", type=Path,
        help="Local directory containing hunyuan3d-dit-v2-mv-turbo/config.yaml and model.fp16.safetensors. "
        "Avoids the official snapshot downloader fetching duplicate .ckpt weights.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--octree-resolution", type=int, default=380)
    parser.add_argument("--num-chunks", type=int, default=20000)
    parser.add_argument("--no-flashvdm", action="store_true", help="Do not select the official turbo VAE decoder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D shape generation requires CUDA for this project workflow")
    if args.steps <= 0 or args.octree_resolution <= 0 or args.num_chunks <= 0:
        raise ValueError("--steps, --octree-resolution, and --num-chunks must be positive")

    labels = ("front", "left", "back")
    images: dict[str, Image.Image] = {}
    for label in labels:
        path = args.inputs / f"{label}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared Hunyuan input: {path}")
        images[label] = Image.open(path).convert("RGBA")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Input views: {', '.join(str(args.inputs / (label + '.png')) for label in labels)}")
    started = time.perf_counter()
    if args.model_dir:
        model_dir = args.model_dir / "hunyuan3d-dit-v2-mv-turbo"
        config_path = model_dir / "config.yaml"
        weights_path = model_dir / "model.fp16.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "--model-dir must contain hunyuan3d-dit-v2-mv-turbo/config.yaml and model.fp16.safetensors"
            )
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=str(weights_path), config_path=str(config_path), use_safetensors=True,
        )
    else:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv",
            subfolder="hunyuan3d-dit-v2-mv-turbo",
            variant="fp16",
        )
    if not args.no_flashvdm and not args.model_dir:
        pipeline.enable_flashvdm()
    mesh = pipeline(
        image=images,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    mesh.export(args.output)
    metadata = {
        "format": "hunyuan3d-2mv-shape-v1",
        "model": "tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo",
        "input_directory": str(args.inputs.resolve()),
        "input_views": list(labels),
        "seed": args.seed,
        "steps": args.steps,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "flashvdm": not args.no_flashvdm and not bool(args.model_dir),
        "output": str(args.output.resolve()),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "elapsed_s": time.perf_counter() - started,
        "coordinate_frame": "Hunyuan canonical frame; alignment to the RGB-D anchor camera is still required.",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"mesh: {args.output} ({metadata['vertices']} vertices, {metadata['faces']} faces)")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
