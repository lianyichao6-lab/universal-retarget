# Robot deployment migration

This document freezes the workstation-side contract for moving the verified
HUG, reconstruction and L25 pipeline to another robot computer.

## Scope

The first deployment uses the hand-mounted Gemini 335Lg as the HUG anchor and
the head-mounted ZED 2i for coarse global observation. It also uses an L25 right
hand, interactive object selection, operator confirmation and a manually taught
reachable arm seed. MANUS teleoperation and automatic active vision are not part of this MVP.

## Portable reconstruction boundary

Every reconstruction backend must produce:

~~~text
reconstruction/
  object_mesh_anchor.ply
  object_surface_anchor.npz
  reconstruction_metadata.json
~~~

Package the current Hunyuan result:

~~~bash
.venv/bin/python tools/package_reconstruction_result.py \
  --mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
  --surface-pointcloud "$SCENE/hunyuan_hybrid_pointcloud.npz" \
  --backend hunyuan3d2-multiview \
  --anchor-frame hand_camera_color_optical_frame \
  --output-dir "$SCENE/deployment/reconstruction"
~~~

A future custom reconstruction model replaces only the two input paths and
backend name. HUG continues to consume object_surface_anchor.npz and contact or
collision planning consumes object_mesh_anchor.ply.

`3DGenerationPipeline` now has a dedicated importer that performs metric-unit
validation, object-to-anchor alignment, mesh surface sampling, measured RGB-D
preservation and packaging in one command. See
`docs/3dgeneration_reconstruction_adapter.md`. Keep Hunyuan as the regression
backend until both backends have been compared on the same captured scenes.

## Portable arm and hand boundary

Export the final collision-aware L25 plan:

~~~bash
.venv/bin/python tools/export_grasp_execution_plan.py \
  --plan "$FINAL_PLAN" \
  --object-mesh "$SCENE/deployment/reconstruction/object_mesh_anchor.ply" \
  --reconstruction-result \
    "$SCENE/deployment/reconstruction/reconstruction_metadata.json" \
  --anchor-frame hand_camera_color_optical_frame \
  --candidate-id "$BEST" \
  --output "$SCENE/deployment/grasp_execution_plan.npz"
~~~

Do not specify pregrasp-offset-hand-m on the workstation. Define it only after
the arm has been manually taught near the object and the L25 local axes have
been verified on the target robot.

The contract provides T_anchor_l25_hand and L25 qpos. The target robot still
must supply:

~~~text
T_robot_base_anchor
T_arm_flange_l25_hand
arm IK and collision checking
operator confirmation and emergency stop
~~~

The flange request is:

~~~text
T_robot_base_arm_flange =
  T_robot_base_anchor
  @ T_anchor_l25_hand
  @ inverse(T_arm_flange_l25_hand)
~~~

## Environment migration

Clone repositories and check out the manifest commits. Recreate Python virtual
environments and rebuild ROS 2 packages on the target. Do not copy .venv,
.venv-hunyuan3d or luban_framework/install between computers.

The HUG checkout intentionally stays on its upstream commit. Reproduce the
workspace integration by applying the tracked patch from the AnyDexRetarget
root:

~~~bash
HUG_BASE=$(cat deployment/patches/hug-base-commit.txt)
git -C external/hug checkout "$HUG_BASE"
git -C external/hug apply "$PWD/deployment/patches/hug-workspace.patch"

# Install after the CUDA-compatible torch packages documented in
# docs/hug_retargeting.md.
.venv/bin/pip install -r deployment/patches/hug-requirements-local.txt
~~~

`external/hug/data/custom/custom.pkl` is generated input data and is not part
of the patch. The target machine must capture or copy its own scene data.

Required large assets include the HUG checkpoint, Hunyuan weights and the
DINOv2 Hugging Face cache. Store them outside Git and verify their hashes with
the deployment manifest.

Generate the manifest on the source workstation after committing all code:

~~~bash
.venv/bin/python tools/create_deployment_manifest.py \
  --workspace /home/evolabs-5080/lianyichao \
  --repo universal-retarget=/home/evolabs-5080/lianyichao/AnyDexRetarget \
  --repo luban=/home/evolabs-5080/lianyichao/luban_framework \
  --repo hunyuan-source=external/hunyuan3d2 \
  --repo somehand-source=external/somehand \
  --asset hug-patch=deployment/patches/hug-workspace.patch \
  --asset hug-requirements=deployment/patches/hug-requirements-local.txt \
  --asset hug-checkpoint=external/hug/checkpoints/hug_full.safetensors \
  --asset hug-mano-assets=external/hug/assets \
  --asset hunyuan-model=external/hunyuan3d2_models/hunyuan3d-dit-v2-mv-turbo \
  --asset dinov2-cache=/home/evolabs-5080/.cache/huggingface/hub/models--facebook--dinov2-with-registers-base \
  --asset cup-regression=outputs/reconstruction/cup_session_run1/deployment \
  --output outputs/deployment/deployment_manifest.json
~~~

On the target machine, provide overrides for assets outside its workspace:

~~~bash
python3 tools/check_deployment_manifest.py \
  --manifest /path/to/copied/deployment_manifest.json \
  --workspace /path/to/workspace \
  --asset dinov2-cache=/target/cache/models--facebook--dinov2-with-registers-base \
  --verify-hashes \
  --require-clean
~~~

## Target-machine order

1. Fill config/deployment/zed2i_gemini335lg_l25.template.yaml.
2. Bring up the head ZED 2i, hand Gemini, arm and L25 independently.
3. Calibrate robot-base to the ZED, arm-flange to the Gemini and arm-flange to the L25 hand base.
4. Validate the point-cloud overlay in robot_base.
5. Teach a reachable pregrasp seed by hand.
6. Transform T_anchor_l25_hand into an arm flange target and run IK only.
7. Preview arm, hand and object together without hardware motion.
8. Execute arm-to-pregrasp with the L25 open.
9. Preshape the L25, approach slowly, close, hold and perform a small lift.
10. Save commanded and measured arm/hand states for every attempt.


## Local Luban bridge simulation

Capture the hand-mounted Gemini with the robot base frame so the saved anchor
view includes `anchor_pose.npz` at the RGB-D timestamp:

~~~bash
/usr/bin/python3 tools/capture_orbbec_rgbd.py \
  --preview --timeout 0 \
  --base-frame r_base_link \
  --camera-frame hand_camera_color_optical_frame \
  --output outputs/deployment_runs/run_001/view_000
~~~

After the normal reconstruction, HUG and L25 planning flow exports
`grasp_execution_plan.npz`, create a calibration-resolved Luban request. The
base-anchor file must be the matching `anchor_pose.npz`; never substitute the
current arm pose after the arm has moved.

~~~bash
.venv/bin/python tools/prepare_luban_grasp_request.py \
  --grasp-contract grasp_execution_plan.npz \
  --base-anchor-capture view_000/anchor_pose.npz \
  --flange-hand calibration/flange_to_l25_hand.npz \
  --pregrasp-offset-hand-m 0 0 -0.10 \
  --output luban_grasp_request.npz
~~~

Run Luban locally with:

~~~bash
cd /home/evolabs-5080/lianyichao/luban_framework
source /opt/ros/jazzy/setup.bash
# Run Luban's supported full build first. A partially copied install tree is not
# sufficient because launch_onboard_nodes also needs the robot descriptions,
# controllers, planner and program packages.
./scripts/build.sh
source .ws/devel/setup.bash
LUBAN_FRAMEWORK_ROOT=$PWD \
LUBAN_ROBOT_MODEL=dual_arm_ar508_l20 RIGHT_HAND_MODEL=l25 LEFT_HAND_MODEL=l25 MOCK=1 NO_PICO=1 RVIZ=1 \
  ros2 launch luban_bringup launch_onboard_nodes.py
~~~

If the Framework is run inside its supported Docker container, the same
workspace root is `/evo`; set `LUBAN_FRAMEWORK_ROOT=/evo` there. `MOCK=1`
selects `mock_components/GenericSystem` and must be retained for this local
validation. Do not use a partially copied `install/` tree as an overlay.

The default executor mode is a dry-run;
the `preview` stage only publishes `/anydex/right_l25_wrist_goal` for RViz.

~~~bash
/usr/bin/python3 tools/luban_ros_grasp_execute.py \
  --request luban_grasp_request.npz

/usr/bin/python3 tools/luban_ros_grasp_execute.py \
  --request luban_grasp_request.npz \
  --stage pregrasp --execute --confirm AR5_L25_CLEAR
~~~

Use `approach` and `close` as separate operator-confirmed stages before `all`.
Luban receives a `PoseStamped` on `/right_arm/target_pose` and exactly 16
active L25 joint radians on `/right_hand_controller/commands`; the internal
21-DoF L25 qpos is not published directly.
