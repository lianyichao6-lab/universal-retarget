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
- `safe_close_trajectory.pkl`: a 16-frame linear trajectory from the O30 Vector neutral posture to
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


The trajectory starts from the collision-free Vector neutral qpos, not from an
unverified physical open pose. Read the real O30 feedback before commanding it
and verify that this neutral pose is appropriate for the mounted hand.

## Standalone O30 MuJoCo

This uses AnyDexRetarget's O30 URDF directly. It does not start a ROS graph.

```bash
.venv/bin/python tools/o30_mujoco_playback.py \
  --trajectory outputs/grasp_o30/candidates/candidate_001/safe_close_trajectory.pkl \
  --fps 10 --no-loop
```

Use `--dry-run` to check the 20-DoF mapping without opening a MuJoCo window.

## Standalone O30 HOP Validation

The direct HOP driver is independent of Luban. For this workstation it is at:

```text
```

First read the hand's current 20 joint feedback without motion. Supply the
actual transport and channel; do not reuse a channel that failed to initialize.

```bash
DRIVER=/path/to/linker_hand_o30_control.py
TRAJECTORY=outputs/grasp_o30/candidates/candidate_001/safe_close_trajectory.pkl

.venv/bin/python tools/o30_hardware_validate.py \
  --trajectory "$TRAJECTORY" \
  --driver "$DRIVER" \
  --comm-type libcanbus --canfd-device 0 --channel <confirmed-channel> \
  --read-state
```

Only after the hand is clear, its feedback and neutral pose have been checked,
and the same safe trajectory has passed MuJoCo, play it at a conservative rate:

```bash
.venv/bin/python tools/o30_hardware_validate.py \
  --trajectory "$TRAJECTORY" --all-frames --fps 10 \
  --driver "$DRIVER" \
  --comm-type libcanbus --canfd-device 0 --channel <confirmed-channel> \
  --execute --confirm O30_HAND_CLEAR
```

This uses HOP's documented 0..255 position values in the audited 20-joint
order. It is hand-only and cannot move either arm.
