# O30 standalone pipeline

This pipeline is independent of Luban and does not generate arm commands.

## Install HUG

```bash
uv pip install --python .venv/bin/python --no-deps -e external/hug
export HUG_CHECKPOINT=/path/to/hug_full.safetensors
```

The tracked HUG source includes the project point-cloud changes and MANO
right-hand asset. Checkpoints and captured data remain local.

## Calibrate O30 scale

```bash
.venv/bin/python tools/calibrate_o30_scale_profile.py \
  --output outputs/calibration/o30_scale_profile.json
```

For guided calibration, pass an NPZ containing `keypoints` or
`human_keypoints_canonical` with shape `N x 21 x 3`.

The O30 profile overrides the 15 Vector scales from measured O30 neutral
finger lengths. Object scale stays fixed at `1.0`; it is never used to hide a
bad hand/object alignment.

## Candidate to hardware gate

Pass `--o30-scale-profile` to HUG candidate generation and rigid refinement.
Use `tools/validate_o30_mesh_plan.py` after refinement. It checks the complete
object surface, all O30 links, cross-finger self-collision, non-pad contact,
and the complete open-to-closed path. A deterministic collision proxy is used
for broad phase, while dense surface samples and final mesh collision remain.

Only `validation_passed: true` plans can be exported:

```bash
.venv/bin/python tools/export_o30_validated_plan_bundle.py \
  --plan outputs/.../o30_rigid_plan.npz \
  --validation outputs/.../full_mesh_validation.json \
  --output outputs/.../o30_grasp_bundle.npz
```

Hardware hand motion is separately calibrated and explicitly gated:

```bash
.venv/bin/python tools/o30_hardware_calibrate.py \
  --output outputs/calibration/o30_hardware_profile.json
.venv/bin/python tools/o30_hardware_execute.py \
  --bundle outputs/.../o30_grasp_bundle.npz \
  --calibration-profile outputs/calibration/o30_hardware_profile.json \
  --execute --confirm O30_HAND_CLEAR
```

No Luban, arm, or ROS planning code is changed by this standalone pipeline.
