# Verified CLI Reference

Status date: 2026-08-20

All commands in this document run from the repository root:

```bash
cd /path/to/universal-retarget
```

Use only the workspace Python environment:

```bash
.venv/bin/python
```

Do not use a Python executable from `didongtai`.

## 1. Video -> MediaPipe -> L25 -> MuJoCo

Vector:

```bash
.venv/bin/python example/teleop_sim.py \
  --video example/data/right.mp4 \
  --show-video \
  --robot l25 \
  --optimizer vector \
  --hand right \
  --no-loop
```

Adaptive:

```bash
.venv/bin/python example/teleop_sim.py \
  --video example/data/right.mp4 \
  --show-video \
  --robot l25 \
  --optimizer adaptive \
  --hand right \
  --no-loop
```

This is the verified continuous demonstration pipeline:

```text
MP4 -> MediaPipe 21x3 -> AnyDexRetarget -> L25 qpos -> MuJoCo
```

## 2. Video Skeleton Comparison

```bash
.venv/bin/python example/test/debug_skeleton.py \
  --input video \
  --video example/data/right.mp4 \
  --robot l25 \
  --optimizer vector \
  --hand right \
  --show-video
```

The viewer overlays raw human skeleton, scaled targets, L25 FK skeleton, and
the L25 model. Change `--optimizer vector` to `--optimizer adaptive` for the
other optimizer.

## 3. Prepare an HUG RGB-D Input

Required files under `external/hug/data/custom/`:

```text
rgb.png
depth.png          uint16, millimeters, registered to RGB
intrinsics.txt     fx fy cx cy, or a 3x3 matrix
```

Prepare `custom.pkl`:

```bash
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m hug.prepare_inputs \
  --dataset-path external/hug/data/custom
```

## 4. RGB-D + Object Click -> HUG Prediction

```bash
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m hug.app \
  --checkpoint-path external/hug/checkpoints/hug_full.safetensors \
  --dataset-path external/hug/data/custom \
  --sample-name custom \
  --port 8081 \
  --save-pred
```

Open the URL printed by Viser, normally `http://localhost:8081`, and click the
target object in the RGB image. Each click writes a new file:

```text
external/hug/data/custom/grasp_pred/custom_<timestamp>.pkl
```

The prediction contains HUG MANO output and `grasp.landmarks_3d` with shape
`(21, 3)`.

## 5. HUG Prediction -> L25 Trajectory

Vector:

```bash
.venv/bin/python tools/hug_static_retarget.py \
  --prediction external/hug/data/custom/grasp_pred/custom_20260820_170049_628.pkl \
  --optimizer vector \
  --output outputs/l25/custom_170049_vector.pkl \
  --frames 60
```

Adaptive:

```bash
.venv/bin/python tools/hug_static_retarget.py \
  --prediction external/hug/data/custom/grasp_pred/custom_20260820_170049_628.pkl \
  --optimizer adaptive \
  --output outputs/l25/custom_170049_adaptive.pkl \
  --frames 60
```

This tool currently targets L25 only. Do not pass `--robot`; that option does
not exist yet. The 60 frames repeat one static HUG grasp for playback.

## 5a. One-command RGB-D Point -> HUG -> L25

The verified non-interactive pipeline is:

```bash
.venv/bin/python tools/grasp_object.py \
  --rgb external/hug/data/custom/rgb.png \
  --depth external/hug/data/custom/depth.png \
  --intrinsics external/hug/data/custom/intrinsics.txt \
  --point 478 285 \
  --robot l25 \
  --optimizer vector \
  --output outputs/grasp/red_cup \
  --dry-run
```

`--point U V` uses pixels in the original RGB image. HUG center-crops a
landscape image to a square before resizing it to 224x224. For the current
672x376 sample, valid original-image points have `u=148..523` and `v=0..375`;
the earlier example point `640 360` is outside the model input and is rejected.

The output directory contains:

```text
prediction.pkl     HUG MANO grasp and 21x3 landmarks
canonical_grasp.npz robot-independent MANO-preserving canonical grasp state
trajectory.pkl     60-frame static trajectory compatible with RViz playback
trajectory.npz     named qpos, keypoints, fingertip FK, solver cost/timing
metadata.json      input/crop/point/limit/dry-run metadata
target_point.png   original RGB with crop boundary and selected point
```

Use `--optimizer adaptive` and a different output directory to run Adaptive.
This command is offline only and sends zero hardware commands.

## 5b. HUG Prediction -> Canonical Grasp State

Use this when HUG was run interactively and you want the canonical intermediate
state without retargeting it immediately:

```bash
.venv/bin/python tools/refine_hug_prediction.py \
  --prediction external/hug/data/custom/grasp_pred/custom_20260820_170049_628.pkl \
  --output outputs/grasp/custom_170049_canonical.npz \
  --hand right
```

The archive is robot-independent and contains named HUG camera-frame 21x3
points, wrist-local canonical 21x3 points, bone lengths, fingertip and pinch
geometry, MANO pose/shape/transformation, the 778-vertex MANO mesh, clicked
RGB-D object point, and fingertip-to-object distances. It preserves the HUG
prediction; it does not optimize contact or alter the grasp.

The one-command pipeline in section 5a writes the same artifact automatically
as `outputs/grasp/<name>/canonical_grasp.npz`.

Retarget a saved canonical state directly; this is the same robot-independent
intermediate that a future contact refinement will update:

```bash
.venv/bin/python tools/hug_static_retarget.py \
  --canonical-grasp outputs/grasp/red_cup/candidates/candidate_004/canonical_grasp.npz \
  --optimizer vector \
  --output outputs/l25/candidate_004_from_canonical_vector.pkl \
  --frames 60
```

## 5c. Object Mask -> Object Point Cloud

Create and manually correct the target-object mask:

```bash
.venv/bin/python tools/create_object_mask.py \
  --rgb external/hug/data/custom/rgb.png \
  --output outputs/grasp/red_cup/object_mask.png
```

Click the target, draw its rectangle, then refine the initial GrabCut mask:
left mouse marks object, right mouse marks background, `g` recomputes, and
Enter saves. Back-project the accepted mask with registered depth:

```bash
.venv/bin/python tools/object_mask_to_pointcloud.py \
  --rgb external/hug/data/custom/rgb.png \
  --depth external/hug/data/custom/depth.png \
  --intrinsics external/hug/data/custom/intrinsics.txt \
  --mask outputs/grasp/red_cup/object_mask.png \
  --output outputs/grasp/red_cup/object_pointcloud.npz
```

Inspect the visible object surface at `http://localhost:8082`:

```bash
.venv/bin/python tools/view_object_pointcloud.py \
  --pointcloud outputs/grasp/red_cup/object_pointcloud.npz
```

## 5d. HUG Multi-candidate Ranking -> L25

Generate ten stochastic HUG candidates with one model load, reject candidates
that are clearly detached from the visible object, rank L25 feasibility, and
save each candidate trajectory:

```bash
.venv/bin/python tools/generate_hug_candidates.py \
  --rgb external/hug/data/custom/rgb.png \
  --depth external/hug/data/custom/depth.png \
  --intrinsics external/hug/data/custom/intrinsics.txt \
  --pointcloud outputs/grasp/red_cup/object_pointcloud.npz \
  --candidates 10 \
  --seed-start 100 \
  --robot l25 \
  --optimizer vector \
  --output outputs/grasp/red_cup/candidates \
  --dry-run
```

The target pixel is read from the mask metadata embedded through the point
cloud, preventing HUG and the mask from silently referring to different
objects. `candidates.csv` and `best_candidate.json` contain a provisional
ranking. Each candidate remains the complete MANO grasp sampled by HUG. The
single-view point cloud is only a weak rejection signal; it does not reward
placing every fingertip on the visible surface. The result is not a
force-closure, hidden-surface collision, or physical-stability metric.

Compare the top three MANO candidates with the object point cloud at
`http://localhost:8083`:

```bash
.venv/bin/python tools/view_hug_candidates.py \
  --candidates-dir outputs/grasp/red_cup/candidates \
  --pointcloud outputs/grasp/red_cup/object_pointcloud.npz \
  --top-k 3
```

Play the generated Vector trajectory in RViz:

```bash
ros2 launch launch/l25_rviz_playback.launch.py \
  trajectory:=outputs/grasp/red_cup/trajectory.pkl \
  fps:=30.0
```

## 6. HUG Skeleton -> L25 MuJoCo Comparison

```bash
.venv/bin/python tools/debug_hug_skeleton.py \
  --prediction external/hug/data/custom/grasp_pred/custom_20260820_170049_628.pkl \
  --robot l25 \
  --optimizer vector \
  --hand right
```

This viewer shows the HUG skeleton, scaled retargeting target, L25 FK skeleton,
and L25 model. Change `--optimizer` to compare Vector and Adaptive.

The HUG/canonical skeleton is robot-independent. The viewer green skeleton
is generated later from L25 robot-specific scaling configuration, so it is
not the canonical grasp state and must be regenerated for another robot.

## 6a. L25 Canonical Calibration Evaluation

Compare the original HUG canonical grasp and its conservative L25-adapted
version with both optimizers. This is offline only and does not send hardware
commands:

```bash
.venv/bin/python tools/evaluate_l25_canonical.py \
  --canonical outputs/grasp/red_cup/candidates/candidate_004/canonical_grasp.npz \
  --adapted outputs/grasp/red_cup/candidate_004_l25_adapted.npz \
  --output outputs/grasp/red_cup/l25_evaluation \
  --overwrite
```

It writes `summary.csv` and `summary.json` with solve timing, joint-limit
state, saturation, thumb-index distance, and Vector-only red-FK-to-green-target
error. `solver_cost` is comparable only between states using the same optimizer;
Vector and Adaptive use different objectives.

## 7. Play an L25 Trajectory in RViz

```bash
ros2 launch launch/l25_rviz_playback.launch.py \
  trajectory:=outputs/l25/custom_170049_vector.pkl \
  fps:=30.0
```

This launch is offline only. It starts no LinkerHand SDK, hardware controller,
or actuator command publisher.

## Verified Artifacts

The custom HUG sample used in the commands above has been verified:

```text
external/hug/data/custom/grasp_pred/custom_20260820_170049_628.pkl
  landmarks_3d: (21, 3)
  finite: true

outputs/l25/custom_170049_vector.pkl
  frames: 60
  MuJoCo qpos shape: (60, 21)
  finite: true
  independent L25 joints published in RViz: 16
```

## Not Implemented Yet

The following are planned interfaces, not working CLIs:

```text
tools/retarget_video.py
tools/benchmark_retargeting.py
--instruction "grasp the red cup"
automatic VLM / grounding / segmentation
contact-aware MANO refinement
O6, L6, and G20 end-to-end retargeting
real LinkerHand command transmission
```

Do not document these as verified until their implementations and tests exist.


## 10. Current L25 Hunyuan + HUG + Four-Backend Pipeline

This is the current verified static-grasp pipeline. `SCENE` identifies a
capture session. Set `BEST` only after the backend benchmark finishes.

```text
Gemini 335 multi-view RGB-D
  -> Hunyuan3D-2mv photo-aligned mesh
  -> visible + generated hybrid point cloud
  -> 50 HUG MANO / CanonicalGraspState candidates
  -> Vector / Adaptive / DexPilot / JointAngle
  -> object-relative L25 optimization
  -> collision-aware refinement and final ranking
  -> MuJoCo -> bounded L25 hardware command
```

Assuming the multi-view capture, masks, aligned mesh, and hybrid cloud have
already been prepared under `SCENE`:

```bash
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

cat "$SCENE/backend_benchmark_50/backend_benchmark.json"
```

After reading the benchmark JSON, select one backend and candidate. The
verified cup run selected `Vector + candidate_017`:

```bash
BACKEND=vector
BEST=candidate_017
PLAN="$SCENE/backend_benchmark_50/$BACKEND/$BEST/l25_collision_aware_plan.npz"
TRAJECTORY="outputs/l25/${BACKEND}_${BEST}_collision_aware.pkl"

.venv/bin/python tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --output-dir "$SCENE/backend_benchmark_50/$BACKEND/$BEST/mujoco_scene" \
  --show

.venv/bin/python tools/l25_plan_to_trajectory.py \
  --plan "$PLAN" --output "$TRAJECTORY" --frames 60

.venv/bin/python tools/l25_mujoco_playback.py \
  --trajectory "$TRAJECTORY" --fps 30 --no-loop
```

For read-only hardware preflight, install the LinkerHand SDK separately and
point to the directory containing `LinkerHand/linker_hand_api.py`:

```bash
export LINKERHAND_SDK_PACKAGE=/path/to/linker_hand_ros2_sdk/linker_hand_ros2_sdk

/usr/bin/python3 tools/l25_hardware_execute.py \
  --sdk-package "$LINKERHAND_SDK_PACKAGE" \
  --trajectory "$TRAJECTORY" --frame 0 --read-state \
  --report "outputs/l25/${BACKEND}_${BEST}_preflight.json"
```

Real motion additionally requires an already configured CAN interface, a clear
physical workspace, `--hardware`, and `--confirm L25_RIGHT_CLEAR`.

Current limitations:

- The contact extractor uses fingertip proximity anchors, not MANO contact
  patches or L25 link contact patches.
- At least three near-surface fingertips are currently required only because
  the rigid alignment uses three points. This is not a physical grasp rule and
  incorrectly rejects valid two-finger pinches.
- Collision refinement handles L25-object penetration but does not yet penalize
  cross-finger self-collision.
- Hybrid Hunyuan geometry is a learned prior and is not guaranteed to improve
  HUG over the measured single-view point cloud.
- The 60 output frames repeat one static qpos; this is not a continuous grasp
  trajectory.
