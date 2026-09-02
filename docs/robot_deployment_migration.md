# Robot deployment migration

This document freezes the workstation-side contract for moving the verified
HUG, reconstruction and L25 pipeline to another robot computer.

## Scope

The first deployment uses the waist Gemini 335Lg as the HUG anchor, the head
Gemini 335Lg as an optional supplementary view, an L25 right hand, interactive
object selection, operator confirmation and a manually taught reachable arm
seed. MANUS teleoperation and automatic active vision are not part of this MVP.

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
  --anchor-frame waist_camera_color_optical_frame \
  --output-dir "$SCENE/deployment/reconstruction"
~~~

A future custom reconstruction model replaces only the two input paths and
backend name. HUG continues to consume object_surface_anchor.npz and contact or
collision planning consumes object_mesh_anchor.ply.

## Portable arm and hand boundary

Export the final collision-aware L25 plan:

~~~bash
.venv/bin/python tools/export_grasp_execution_plan.py \
  --plan "$FINAL_PLAN" \
  --object-mesh "$SCENE/deployment/reconstruction/object_mesh_anchor.ply" \
  --reconstruction-result \
    "$SCENE/deployment/reconstruction/reconstruction_metadata.json" \
  --anchor-frame waist_camera_color_optical_frame \
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

Required large assets include the HUG checkpoint, Hunyuan weights and the
DINOv2 Hugging Face cache. Store them outside Git and verify their hashes with
the deployment manifest.

Generate the manifest on the source workstation after committing all code:

~~~bash
.venv/bin/python tools/create_deployment_manifest.py \
  --workspace /home/evolabs-5080/lianyichao \
  --repo universal-retarget=/home/evolabs-5080/lianyichao/AnyDexRetarget \
  --repo luban=/home/evolabs-5080/lianyichao/luban_framework \
  --asset hug-checkpoint=external/hug/checkpoints/hug_full.safetensors \
  --asset hunyuan-model=external/hunyuan3d2_models/hunyuan3d-dit-v2-mv-turbo \
  --asset dinov2-cache=/home/evolabs-5080/.cache/huggingface/hub/models--facebook--dinov2-with-registers-base \
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

1. Fill config/deployment/dual_gemini335lg_l25.template.yaml.
2. Bring up head camera, waist camera, arm and L25 independently.
3. Calibrate robot-base to both cameras and arm-flange to L25 hand base.
4. Validate the point-cloud overlay in robot_base.
5. Teach a reachable pregrasp seed by hand.
6. Transform T_anchor_l25_hand into an arm flange target and run IK only.
7. Preview arm, hand and object together without hardware motion.
8. Execute arm-to-pregrasp with the L25 open.
9. Preshape the L25, approach slowly, close, hold and perform a small lift.
10. Save commanded and measured arm/hand states for every attempt.
