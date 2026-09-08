# O30 Full Mesh Pipeline

This is the O30 equivalent of the final L25 offline grasp-planning chain. It
uses the already captured cup RGB-D inputs and photo-aligned mesh. It does not
start Luban and never commands hardware.

## 1. Produce HUG candidates

```bash
cd /home/evolabs-5080/lianyichao/AnyDexRetarget
SCENE=outputs/reconstruction/cup_session
OUT=$SCENE/o30_hug_candidates_50

.venv/bin/python tools/generate_hug_candidates.py \
  --rgb "$SCENE/view_000/rgb.png" \
  --depth "$SCENE/view_000/depth.png" \
  --intrinsics "$SCENE/view_000/intrinsics.txt" \
  --pointcloud "$SCENE/view_000/object_pointcloud.npz" \
  --hug-pointcloud "$SCENE/hunyuan_hybrid_pointcloud.npz" \
  --robot o30 --optimizer vector --candidates 50 \
  --output "$OUT" --dry-run
```

The HUG ranking is only a proposal ranking. Do not choose a physical O30 grasp
from that score alone: the O30 thumb geometry differs from L25.

## 2. O30 final contact trials

For each promising HUG candidate, fit the O30 distal pads to the mesh anchors.
The planner first uses a no-scale rigid alignment, then allows a bounded
O30-specific wrist/object-relative correction. This retains the object metric
scale and HUG contact intent while resolving morphology differences between a
human/L25 hand and O30.

```bash
CAND=$OUT/candidate_028
TRIAL=$CAND/o30_wrist_compensated_trials/thumb_index_middle
mkdir -p "$TRIAL"

.venv/bin/python tools/extract_object_relative_contacts.py \
  --canonical-grasp "$CAND/canonical_grasp.npz" \
  --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
  --near-surface-gap-mm 30 \
  --output "$TRIAL/contact_plan.npz"

.venv/bin/python tools/plan_o30_rigid_object_relative_grasp.py \
  --contact-plan "$TRIAL/contact_plan.npz" \
  --contact-fingers thumb,index,middle \
  --surface-gap-mm 0 \
  --wrist-translation-limit-mm 60 \
  --wrist-rotation-limit-deg 35 \
  --output "$TRIAL/o30_wrist_compensated_plan.npz"

.venv/bin/python tools/build_o30_object_relative_scene.py \
  --plan "$TRIAL/o30_wrist_compensated_plan.npz" \
  --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
  --output-dir "$TRIAL/scene"

.venv/bin/python tools/refine_o30_collision_aware.py \
  --plan "$TRIAL/o30_wrist_compensated_plan.npz" \
  --scene-xml "$TRIAL/scene/o30_object_relative_scene.urdf" \
  --mesh-contact-weight 12 \
  --max-joint-delta-rad 0.35 \
  --max-iterations 180 \
  --output "$TRIAL/o30_wrist_mesh_closed_plan.npz"
```

The final refinement minimizes fingertip distance to the *displayed mesh*,
while rejecting object contact from the palm/proximal links and cross-finger
penetration. It is still a geometry diagnostic, not a force-closure proof.

## 3. Rank and inspect the static final pose

```bash
.venv/bin/python tools/rank_o30_object_relative_candidates.py \
  --candidates-dir "$OUT"

.venv/bin/python tools/o30_mujoco_playback.py \
  --trajectory "$TRIAL/o30_wrist_mesh_closed_plan_trajectory.pkl" \
  --scene-xml "$TRIAL/scene/o30_object_relative_scene.urdf" \
  --final-pose
```

The current verified cup trial is `candidate_028`, using thumb/index/middle.
It has final actual-mesh distances of 0.163 mm, 0.083 mm, and 0.016 mm,
respectively, with zero reported forbidden object or cross-finger penetration.

For a 50-candidate batch, run the section 2 commands for the desired contact
patterns (for example `thumb,index,middle`, `thumb,index,ring`, and
`thumb,index,pinky`) under each candidate, then run the ranker. The ranker
first requires collision safety and at least three active contacts, then ranks
by actual displayed-mesh distance rather than HUG target-point error.

The mesh and hand share an O30 simulation hand frame. This is intentionally
independent from Luban. Before using any independent O30 hardware driver, a
human must verify the static viewer, native neutral pose, joint directions,
scale, CAN channel, camera-to-hand calibration, and clear workspace. Offline
mesh collision results do not authorize physical actuation.
