[中文](README.zh.md) | English

# Universal Retarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#current-status)

Universal Retarget is a research workspace for converting human-level grasp intent into executable poses for heterogeneous dexterous hands. The verified path combines RGB-D perception, learned 3D shape completion, HUG MANO grasp generation, a robot-independent canonical hand state, multiple retargeting backends, object-relative L25 optimization, collision-aware refinement, MuJoCo inspection, and bounded LinkerHand hardware execution.

The project is built on [AnyDexRetarget](https://github.com/qqsq12321/AnyDexRetarget). Its complete history is preserved in the [upstream-history](https://github.com/lianyichao6-lab/universal-retarget/tree/upstream-history) branch.

## Verified Pipeline

~~~text
Orbbec Gemini 335 multi-view RGB-D
  -> masks and measured visible point cloud
  -> Hunyuan3D-2mv mesh completion and photo alignment
  -> measured + generated hybrid surface point cloud
  -> HUG: 50 stochastic MANO grasp candidates
  -> CanonicalGraspState
  -> AnyDex Vector | AnyDex Adaptive | DexPilot | JointAngle
  -> object-relative L25 qpos optimization
  -> object-collision refinement and final L25 ranking
  -> MuJoCo inspection
  -> bounded 0..255 LinkerHand CAN command
~~~

The cup experiment was validated end to end with **Vector + candidate_017**, including a successful physical pickup. This is an experimental result, not a general grasp-success guarantee.

## What This Repository Adds

- **RGB-D and object geometry**: interactive Gemini 335 capture, masks, measured point clouds, Hunyuan multi-view completion, photo-frame alignment, and hybrid point clouds.
- **HUG integration**: repeated stochastic MANO sampling from one model load, candidate artifacts, visualization, and provisional ranking.
- **Canonical representation**: a robot-independent CanonicalGraspState preserving wrist-local 21 x 3 landmarks, MANO pose and shape, the 778-vertex mesh, transforms, confidence, and object-relative data.
- **Four L25 backends**: AnyDex Vector, AnyDex Adaptive, dex-retargeting DexPilot, and dex-retargeting JointAngle.
- **Object-relative planning**: HUG-derived surface anchors, rigid hand-object alignment, continuous L25 joint optimization, penetration correction, and final reranking.
- **Simulation and hardware**: L25 MuJoCo scenes, static qpos trajectories, full-range 0..255 SDK mapping, read-only preflight, and bounded CAN replay.
- **Robot assets and calibration**: L25, L6, and O6 assets plus L25 geometry-calibrated configurations. The complete path is currently verified only on the right L25.

## Representation Boundary

~~~text
HUG MANO grasp
  -> camera-frame landmarks (21, 3)
  -> CanonicalGraspState
       - wrist-local landmarks (21, 3)
       - MANO pose / shape
       - MANO mesh (778, 3)
       - camera and canonical transforms
       - object point and contact metadata
  -> backend-specific human target
  -> L25 qpos in radians
  -> LinkerHand channel command in 0..255
~~~

**21 x 3** means 21 human-hand landmarks with one XYZ coordinate per landmark. It is not a 21-DOF joint command. Robot qpos is a separate morphology-specific joint-space representation.

## Current Status

| Component | Status |
|---|---|
| Gemini 335 registered RGB-D capture | Verified |
| Multi-view Hunyuan mesh completion | Verified experimentally |
| Hybrid measured/generated point cloud | Implemented |
| HUG MANO candidate generation | Verified |
| CanonicalGraspState | Implemented and used |
| Vector / Adaptive / DexPilot / JointAngle | Compared on L25 |
| Object-relative L25 optimization | Implemented |
| L25-object penetration refinement | Implemented |
| MuJoCo L25 + object inspection | Verified |
| LinkerHand L25 CAN execution | Verified with bounded replay |
| L6 / O6 end-to-end autonomous grasp | Not yet verified |
| VLM language grounding | Planned |
| Arm planning and camera-to-robot calibration | Out of scope |
| Tactile closed-loop recovery | Planned |

## Installation Boundary

~~~bash
git clone git@github.com:lianyichao6-lab/universal-retarget.git
cd universal-retarget
~~~

The repository intentionally excludes large or separately licensed components:

- Python virtual environments
- HUG source and checkpoints
- Hunyuan3D source and model weights
- MANO model assets
- LinkerHand vendor SDK
- RGB-D captures, generated meshes, predictions, and experiment outputs

Base package metadata is in [pyproject.toml](pyproject.toml). Reconstructing the verified workstation environment requires the external components above; consult [the CLI reference](docs/cli_reference.md) before running the pipeline.

## Quick Start From a Prepared Scene

This example starts at a scene containing an anchor RGB-D frame, measured point cloud, aligned Hunyuan mesh, and hybrid point cloud:

~~~bash
SCENE=outputs/reconstruction/object_session_run1

env -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python tools/generate_hug_candidates.py \
  --rgb "$SCENE/view_000/rgb.png" \
  --depth "$SCENE/view_000/depth.png" \
  --intrinsics "$SCENE/view_000/intrinsics.txt" \
  --pointcloud "$SCENE/view_000/object_pointcloud.npz" \
  --hug-pointcloud "$SCENE/hunyuan_hybrid_pointcloud.npz" \
  --robot l25 --optimizer vector \
  --candidates 50 --sampling-steps 50 \
  --frames 60 --fps 30 \
  --output "$SCENE/hug_candidates_50" --dry-run

.venv/bin/python tools/benchmark_l25_retarget_backends.py \
  --candidates-dir "$SCENE/hug_candidates_50" \
  --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
  --output-dir "$SCENE/backend_benchmark_50"
~~~

Read backend_benchmark.json before selecting a backend and candidate:

~~~bash
BACKEND=vector
BEST=candidate_017
PLAN="$SCENE/backend_benchmark_50/$BACKEND/$BEST/l25_collision_aware_plan.npz"

.venv/bin/python tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --output-dir "$SCENE/backend_benchmark_50/$BACKEND/$BEST/mujoco_scene" \
  --show
~~~

For the complete workflow from camera startup through hardware execution, use [docs/cli_reference.md](docs/cli_reference.md). Hardware commands require a clear workspace, installed LinkerHand SDK, active CAN interface, and the explicit executor confirmation token.

## Candidate Ranking

Final selection uses the L25 result, not only the initial HUG score:

- L25 distal-pad target mean and maximum error
- thumb/finger opposition and contact-anchor spread across object sides
- object penetration and cross-finger self-collision
- joint-limit margin and saturation
- deviation from the initial retargeted pose
- optional hardware command-tracking error when a measured report is supplied

The first Viser top-three view helps reject implausible MANO grasps, but does not replace final robot-specific reranking.

The four-backend benchmark shares one MuJoCo collision proxy and one set of
backend-independent contact plans. Use `--limit 10` for a quick smoke test;
omit it for the full candidate set.

## Repository Layout

~~~text
anydexretarget/
  hand_representation.py        CanonicalGraspState
  optimizer/                    AnyDex optimization backends
assets/
  linkerhand_l25/               L25 URDF, MuJoCo, and meshes
  linkerhand_l6/                L6 assets
  linkerhand_o6/                O6 assets
example/config/                 robot/input/optimizer configurations
tools/
  capture_orbbec_rgbd.py        interactive RGB-D capture
  generate_hug_candidates.py    HUG MANO candidate generation
  benchmark_l25_retarget_backends.py
  rerank_l25_object_relative_candidates.py
  refine_l25_collision_aware.py
  build_l25_object_relative_scene.py
  l25_hardware_execute.py
docs/                           verified CLI and integration notes
tests/                          focused representation and pipeline tests
~~~

## Known Limitations

- Hunyuan completion is a learned prior; a hybrid cloud is not guaranteed to outperform measured single-view depth.
- Contact targets use three samples along each MANO distal phalanx as a distal-pad proxy; they are not dense MANO pressure patches.
- Two-finger pinches are supported by fitting two contacts plus the wrist reference; three-finger and enveloping grasps use all active contact proxies.
- MuJoCo refinement penalizes object penetration and cross-finger self-collision, but remains dependent on collision-mesh quality.
- Output trajectories repeat one static qpos; approach, closure, force control, and release are not generated.
- The project does not move a robot arm or transform camera-frame objects into a robot base frame.
- There is no force-closure proof, tactile loop, or universal physical grasp guarantee.

## Documentation

- [Verified CLI reference](docs/cli_reference.md)
- [HUG integration boundary](docs/hug_retargeting.md)
- [Orbbec Gemini 335 capture](docs/orbbec_gemini335_capture.md)
- [CAD or mesh input to HUG](docs/cad_mesh_to_hug.md)
- [L25 Pico 4 simulation](docs/l25_pico4_simulation.md)

## Upstream, Branches, and License

- **main**: Universal Retarget project and current documentation.
- **upstream-history**: complete AnyDexRetarget history.
- **upstream**: [qqsq12321/AnyDexRetarget](https://github.com/qqsq12321/AnyDexRetarget).

The repository retains the upstream MIT license in [LICENSE](LICENSE). HUG, Hunyuan3D, MANO, dex-retargeting, LinkerHand assets, and vendor SDK components remain subject to their own licenses.

## Safety

Hardware execution can damage a hand, object, or nearby equipment. Inspect the exact qpos in MuJoCo, run read-only preflight first, clear the workspace, use conservative speed and torque, and keep an emergency stop available.
