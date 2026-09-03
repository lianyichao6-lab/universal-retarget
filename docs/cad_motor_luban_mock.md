# CAD Motor Luban Mock

This is the fixed-object AR5 + L25 mock workflow. It verifies the same object
frame is used by CAD HUG inputs, L25 planning and the Luban scene. It does not
simulate force closure or object lifting.

Use the deployment branch and create three HUG input views of the standard
motor. For each view run `generate_hug_candidates.py` with 10 candidates and
then `benchmark_l25_retarget_backends.py`. Only a candidate marked
`recommended: true` may continue.

```bash
MOTOR=assets/cad/P5636-36A__20250920001
RUN=outputs/cad_motor/p5636_v1
BLENDER=/path/to/blender

.venv/bin/python tools/render_cad_hug_inputs.py \
  --mesh "$MOTOR/model.obj" --output "$RUN/view_00" --blender "$BLENDER" \
  --azimuth-deg 0 --elevation-deg 0
```

Repeat with `view_01` at `--azimuth-deg 45 --elevation-deg -15` and `view_02`
at `--azimuth-deg 90 --elevation-deg 0`. The generated metadata contains the
target pixel for `generate_hug_candidates.py`.

For a selected Vector candidate, export an object-anchored execution contract:

```bash
BEST=candidate_004
.venv/bin/python tools/export_grasp_execution_plan.py \
  --plan "$RUN/backend_benchmark/vector/$BEST/l25_collision_aware_plan.npz" \
  --contact-plan "$RUN/backend_benchmark/_shared/contact_plans/$BEST/contact_plan.npz" \
  --anchor-frame motor_object --candidate-id "$BEST" \
  --output "$RUN/deployment/grasp_execution_plan.npz"
```

Start Luban mock with the original metre-scale CAD mesh. The six pose values
must be manually adjusted in RViz to a reachable table location and reused
unchanged when creating the request.

```bash
export LUBAN_MUJOCO_OBJECT_MESH="$PWD/$MOTOR/model.obj"
export LUBAN_MUJOCO_OBJECT_POSE="X Y Z ROLL PITCH YAW"
```

The motor OBJ has a lower Z bound near `-0.032 m`. With the standard table top
at `-0.51 m`, its center Z begins near `-0.478 m` when its CAD Z axis is
vertical. There is intentionally no default X/Y: teach a reachable mock pose.

Write the matching object anchor and prepare the existing Luban request:

```bash
.venv/bin/python tools/create_static_anchor_pose.py \
  --pose X Y Z ROLL PITCH YAW --anchor-frame motor_object --base-frame world \
  --output "$RUN/deployment/motor_object_pose.npz"

# MOCK=1 only: mirrors Luban's r_hand_mount URDF joint.
# Replace with the measured hand-mount calibration on hardware.
.venv/bin/python tools/create_l25_nominal_mount.py \
  --output "$RUN/deployment/l25_nominal_mount_mock.npz"

.venv/bin/python tools/prepare_luban_grasp_request.py \
  --grasp-contract "$RUN/deployment/grasp_execution_plan.npz" \
  --base-anchor-capture "$RUN/deployment/motor_object_pose.npz" \
  --flange-hand "$RUN/deployment/l25_nominal_mount_mock.npz" \
  --base-frame world --anchor-frame motor_object \
  --pregrasp-offset-hand-m 0 0 -0.10 \
  --output "$RUN/deployment/luban_grasp_request.npz"
```

For hardware, `--flange-hand` must instead point to the measured file with
the NPZ key `T_arm_flange_l25_hand`.

Run `luban_ros_grasp_execute.py` one stage at a time: `preview`, `pregrasp`,
`approach`, then `close`. Keep `MOCK=1`; hardware execution is not part of this
workflow.

## ROS Preview

When Luban runs in its Docker container, it uses localhost-only DDS discovery.
Use the same setting in the AnyDex terminal and preview the wrist target before
any motion stage:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONPATH=".:$PYTHONPATH"

/usr/bin/python3 tools/luban_ros_grasp_execute.py \
  --request "$RUN/deployment/luban_grasp_request.npz" \
  --stage preview --execute --confirm AR5_L25_CLEAR
```

The preview waits for an RViz or `ros2 topic echo` subscriber and publishes only
`/anydex/right_l25_wrist_goal`; it does not command the arm or hand.
