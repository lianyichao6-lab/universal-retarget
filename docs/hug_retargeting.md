# HUG -> AnyDexRetarget

## What is connected

The official [HUG](https://github.com/KevinyWu/hug) release is an
object-conditioned RGB-D grasp predictor. It takes one registered RGB image,
depth image, camera intrinsics, and a selected object pixel, then predicts a
single right-hand MANO grasp. It is not a video hand tracker.

This repository keeps HUG isolated under `external/hug` and feeds its saved
prediction through the existing retargeting core:

```text
RGB-D + object point
  -> HUG
  -> grasp.landmarks_3d (21, 3), camera frame, meters
  -> anydexretarget.hug_adapter
  -> Retargeter (vector or adaptive)
  -> L25 qpos
  -> MuJoCo
```

HUG's landmarks use the same MANO/MediaPipe semantic layout used by this
project: wrist index 0, then thumb/index/middle/ring/pinky chains, with tips at
indices 4, 8, 12, 16, and 20. The adapter validates this contract; it does not
silently reorder or mirror points.

## Prerequisites

HUG is checked out under `external/hug` and installed into this project .venv;
no Python executable from another workspace is required. The large checkpoint
and MANO asset are kept on disk under `external/hug` and are ignored by Git.

For a fresh checkout, install the CUDA wheels first, then the remaining runtime
list in `external/hug/requirements-local.txt`:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128   torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
uv pip install --python .venv/bin/python \
  --find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html   torch-cluster==1.6.3+pt29cu128
uv pip install --python .venv/bin/python \
  -r external/hug/requirements-local.txt
uv pip install --python .venv/bin/python --no-deps -e external/hug
```

The local checkpoint is `external/hug/checkpoints/hug_full.safetensors` and
MANO is under `external/hug/assets/mano_models/`.

## Run one HUG prediction

Prepare an RGB-D sample with `rgb.png`, `depth.png` (uint16 millimeters),
and `intrinsics.txt` under a custom dataset folder. Run from the repository
root:

```bash
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m hug.prepare_inputs \
  --dataset-path external/hug/data/custom
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m hug.app \
  --checkpoint-path external/hug/checkpoints/hug_full.safetensors \
  --dataset-path external/hug/data/custom \
  --save-pred
```

The saved file is under `external/hug/data/custom/grasp_pred/*.pkl`.

## Feed the prediction to L25

The extraction/validation layer can be exercised without importing HUG's heavy
model dependencies:

```bash
../.venv/bin/python -c \
  "from anydexretarget.hug_adapter import load_prediction; \
   p=load_prediction('data/custom/grasp_pred/00000000.pkl'); \
   print(p.keypoints_3d.shape, p.keypoints_3d.dtype, p.keypoints_3d[0])"
```

Use `p.keypoints_3d` as the input to the existing L25 `Retargeter`. The
existing `mediapipe_rotation`, L25 scaling, tip offsets, and joint-limit checks
still apply. HUG landmarks are metric camera-frame points; do not multiply
them by the MediaPipe video depth-scale parameter.

## Important limitation

The current `right.mp4` MediaPipe path remains the correct continuous-video
pipeline. HUG should first be validated on one RGB-D frame and one static L25
pose. Running HUG independently on every video frame would be slow, stochastic,
object-conditioned, and temporally inconsistent. A future video system should
use HUG for an initial grasp prior and a tracker/MediaPipe/Pico input for the
subsequent sequence, or add an explicit temporal optimizer.
