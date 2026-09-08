# O30 HUG Collision Filter

After generating O30 Vector candidates, run the O30 mesh filter before using a
candidate in MuJoCo or sending a hand command. It operates entirely offline and
never creates ROS publishers.

```bash
.venv/bin/python tools/score_o30_hug_candidates.py \
  --candidates-dir outputs/grasp_o30/candidates \
  --pointcloud outputs/grasp/red_cup/object_pointcloud.npz \
  --closure-steps 16 \
  --forbidden-clearance-mm 3 \
  --mesh-vertices-per-link 384
```

The command writes the following inside each candidate directory:

- `o30_collision_metrics.json`: O30 mesh self-collision pairs, per-link
  visible-object clearance, non-tip violations, and fingertip distances.
- `safe_close_trajectory.pkl`: a 16-frame linear trajectory from open O30 to
  the maximum mesh-safe closure fraction. This is suitable for MuJoCo playback
  or inspection, not direct real-hardware authorization.

It also writes `o30_collision_ranking.json` at the candidate root. Ranking is:
full safe closure first, then fewer prohibited object contacts and self
collisions, then the previous HUG/Vector score.

The visible object is a metric RGB-D point cloud in the HUG camera frame. The
filter transforms it into the exact O30 Vector retarget frame before scoring.
The current O30 Vector scale and rotation constants remain initial morphology
fit values; verify them with the physical O30 hand before making contact.

This filter permits distal fingertip contact but rejects a palm or non-distal
link closer than the clearance threshold. It samples O30 collision meshes and
measures distance to the observed point-cloud surface. Therefore it does not
prove force closure, account for unseen object surfaces, replace arm collision
planning, or certify a real grasp. Use it to choose the candidate, then run the
full scene in MuJoCo and use conservative real-hand close/force limits.
