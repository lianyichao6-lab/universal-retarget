#!/usr/bin/env python3
"""Download only the Hunyuan3D-2mv turbo config and safetensors checkpoint.

The official snapshot helper fetches both a ``.ckpt`` and ``.safetensors`` copy
of the same model.  This downloader intentionally keeps one copy so the local
mesh experiment has a bounded disk and network footprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "tencent/Hunyuan3D-2mv"
SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"
FILES = ("config.yaml", "model.fp16.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in FILES:
        relative_path = f"{SUBFOLDER}/{name}"
        path = hf_hub_download(
            repo_id=REPO,
            filename=relative_path,
            local_dir=args.output,
        )
        print(f"downloaded: {path}")


if __name__ == "__main__":
    main()
