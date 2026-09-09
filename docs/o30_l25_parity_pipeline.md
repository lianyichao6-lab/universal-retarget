# O30 L25-Parity Offline Pipeline

This is the authoritative standalone O30 workflow. It uses the same RGB-D scene, metric mesh, and 50 HUG canonical grasps as the verified L25 run. It does not start Luban and it never commands the O30 hand.

## Inputs

```bash
cd /home/evolabs-5080/lianyichao/AnyDexRetarget-core
PYTHON=/home/evolabs-5080/lianyichao/AnyDexRetarget/.venv/bin/python
export PYTHONPATH=$PWD
SCENE=/home/evolabs-5080/lianyichao/AnyDexRetarget/outputs/reconstruction/cup_session_run1
CANDIDATES=$SCENE/hug_candidates_50
MESH=$SCENE/hunyuan_mv_mesh_photo_aligned.ply
OUT=outputs/o30_l25_parity_50
```

`$CANDIDATES` is the common HUG result. Do not regenerate it from a different RGB-D frame, point cloud, or mesh. The human grasp candidates are shared; each O30 backend creates its own hand pose afterward.

## Four-backend O30 ranking

```bash
$PYTHON tools/benchmark_o30_retarget_backends.py \
  --candidates-dir "$CANDIDATES" \
  --object-mesh "$MESH" \
  --output-dir "$OUT"
```

The command processes all 50 candidates with Vector, Adaptive, DexPilot, and JointAngle. For each candidate it creates an O30 distal-pad object-relative plan, refines against object and self collision, then applies the strict full-STL closure gate.

Read the result before inspecting or exporting anything:

```bash
cat "$OUT/backend_benchmark.json"
```

Only a row with `best_recommended: true` may progress. A rejected pose is useful diagnostic output but is not a grasp and must not be sent to hardware.

## MuJoCo final pose

Set `BACKEND` and `BEST` from a recommended row:

```bash
BACKEND=vector
BEST=candidate_000
PLAN="$OUT/$BACKEND/$BEST/o30_collision_aware_plan.npz"
SCENE_XML="$OUT/$BACKEND/$BEST/collision_aware_scene/o30_object_relative_scene.urdf"
TRAJECTORY="$OUT/$BACKEND/$BEST/o30_collision_aware_plan_trajectory.pkl"

$PYTHON tools/o30_bundle_to_trajectory.py \
  --plan "$PLAN" --output "$TRAJECTORY"

$PYTHON tools/o30_mujoco_playback.py \
  --trajectory "$TRAJECTORY" \
  --scene-xml "$SCENE_XML" \
  --final-pose
```

The plan conversion is simulation only. It intentionally does not produce a hardware-approved bundle.

## Hardware gate

A hardware bundle is permitted only after the selected plan passed its strict validation report:

```bash
VALIDATION="$OUT/$BACKEND/$BEST/full_mesh_validation.json"
BUNDLE="$OUT/$BACKEND/$BEST/o30_grasp_bundle.npz"

$PYTHON tools/export_o30_validated_plan_bundle.py \
  --plan "$PLAN" --validation "$VALIDATION" --output "$BUNDLE"
```

This validates the object remains at metric scale, requires thumb plus at least two non-thumb contacts, rejects palm/proximal and cross-finger collision, and checks the full closure path. Independent O30 hardware calibration and explicit execution remain separate safety steps.
