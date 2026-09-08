# O30 Full Mesh Pipeline

This is the O30 equivalent of the L25 offline chain. It uses the already
captured cup RGB-D data and the existing photo-aligned Hunyuan mesh. It does
not recapture images, start Luban, or command hardware.

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

For every candidate, extract HUG-to-mesh contacts, fit O30, build a MuJoCo
scene containing the real reconstructed mesh, then refine forbidden object and
cross-finger collision. Candidates without two mesh-near HUG fingers are
skipped deliberately.

```bash
for CAND in "$OUT"/candidate_*; do
  RUN="$CAND/o30_object_relative"
  mkdir -p "$RUN"

  .venv/bin/python tools/extract_object_relative_contacts.py \
    --canonical-grasp "$CAND/canonical_grasp.npz" \
    --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
    --near-surface-gap-mm 30 --output "$RUN/contact_plan.npz" || continue

  .venv/bin/python tools/plan_o30_object_relative_grasp.py \
    --contact-plan "$RUN/contact_plan.npz" \
    --output "$RUN/o30_object_relative_plan.npz" || continue

  .venv/bin/python tools/build_o30_object_relative_scene.py \
    --plan "$RUN/o30_object_relative_plan.npz" \
    --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
    --output-dir "$RUN/scene" || continue

  .venv/bin/python tools/refine_o30_collision_aware.py \
    --plan "$RUN/o30_object_relative_plan.npz" \
    --scene-xml "$RUN/scene/o30_object_relative_scene.urdf" \
    --output "$RUN/o30_collision_refined_plan.npz" || continue

done
```

Rank completed candidates, then inspect the selected O30 hand and cup together
in the independent MuJoCo viewer:

```bash
.venv/bin/python tools/rank_o30_object_relative_candidates.py \
  --candidates-dir "$OUT"

BEST=candidate_XXX  # Read from $OUT/o30_object_relative_ranking.json
RUN="$OUT/$BEST/o30_object_relative"

.venv/bin/python tools/o30_mujoco_playback.py \
  --trajectory "$RUN/o30_collision_refined_plan_trajectory.pkl" \
  --scene-xml "$RUN/scene/o30_object_relative_scene.urdf" \
  --fps 10 --no-loop
```

The mesh and hand share the O30 simulation hand frame. This is intentionally
separate from Luban. Only after a human reviews this view and verifies O30
neutral posture, joint direction, scale, CAN channel, and clear workspace may
the independent HOP hardware validator be used. The offline collision result
is not a permission to actuate the hand.
