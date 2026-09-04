# L25 手部全链路

本文是手部专用流程：

```text
3DGeneration mesh + RGB-D
  -> 对齐到相机坐标系
  -> HUG 多候选抓取
  -> L25 重定向与碰撞排序
  -> 带物体 mesh 的 MuJoCo
  -> LinkerHand L25 真机预检/有界执行
```

本文不启动 AR5、Luban 或 ROS。

## 0. 准备路径

只修改以下变量：

```bash
export RETARGET_ROOT=/path/to/AnyDexRetarget
export SCENE="$RETARGET_ROOT/outputs/l25_session_001"
export GEN_MESH=/path/to/3dgeneration/object.glb
export T_ANCHOR_OBJECT=/path/to/calibration/T_anchor_object.json
export L25_SDK=/path/to/linkerhand_sdk
export CAN_IFACE=can0
export PY="$RETARGET_ROOT/.venv/bin/python"

cd "$RETARGET_ROOT"
```

`$SCENE/view_000/` 下必须有：

```text
rgb.png
depth.png
intrinsics.txt
object_pointcloud.npz
```

检查环境：

```bash
"$PY" -c "import mujoco; print('MuJoCo', mujoco.__version__)"
test -f "$GEN_MESH"
test -f "$T_ANCHOR_OBJECT"
test -f "$L25_SDK/LinkerHand/linker_hand_api.py"
```

## 1. 导入 3DGeneration mesh

`T_anchor_object` 必须把物体坐标系中的米制点变换到相机坐标系。使用已测矩阵：

```bash
"$PY" tools/import_3dgeneration_reconstruction.py \
  --mesh "$GEN_MESH" \
  --anchor-pointcloud "$SCENE/view_000/object_pointcloud.npz" \
  --transform "$T_ANCHOR_OBJECT" \
  --backend 3dgenerationpipeline-fine \
  --anchor-frame hand_camera_color_optical_frame \
  --output-dir "$SCENE/3dgeneration/reconstruction" \
  --overwrite
```

如果导出 mesh 使用毫米：

```bash
  --source-unit mm
```

确认输出：

```text
$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply
$SCENE/3dgeneration/reconstruction/object_surface_anchor.npz
$SCENE/3dgeneration/reconstruction/reconstruction_metadata.json
```

## 2. 检查 mesh 对齐

```bash
"$PY" tools/project_mesh_to_rgb.py \
  --mesh "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --rgb "$SCENE/view_000/rgb.png" \
  --intrinsics "$SCENE/view_000/intrinsics.txt" \
  --output "$SCENE/3dgeneration/reconstruction/alignment_overlay.png"
```

必须人工确认 mesh 与真实物体的位置、尺度和朝向重合。错位时先修正标定矩阵。

## 3. 生成 HUG 候选

```bash
export HUG_OUTPUT="$SCENE/hug_candidates_50"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" tools/generate_hug_candidates.py \
  --rgb "$SCENE/view_000/rgb.png" \
  --depth "$SCENE/view_000/depth.png" \
  --intrinsics "$SCENE/view_000/intrinsics.txt" \
  --pointcloud "$SCENE/view_000/object_pointcloud.npz" \
  --hug-pointcloud "$SCENE/3dgeneration/reconstruction/object_surface_anchor.npz" \
  --robot l25 --optimizer vector \
  --candidates 50 --sampling-steps 50 \
  --frames 60 --fps 30 \
  --output "$HUG_OUTPUT" --dry-run
```

## 4. L25 碰撞排序

```bash
export BENCHMARK_OUTPUT="$SCENE/backend_benchmark_50"

"$PY" tools/benchmark_l25_retarget_backends.py \
  --candidates-dir "$HUG_OUTPUT" \
  --object-mesh "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --output-dir "$BENCHMARK_OUTPUT"

cat "$BENCHMARK_OUTPUT/backend_benchmark.json"
```

选择排序结果中的后端和候选：

```bash
export BACKEND=vector
export BEST=candidate_017
export PLAN="$BENCHMARK_OUTPUT/$BACKEND/$BEST/l25_collision_aware_plan.npz"
export TRAJECTORY="$SCENE/${BACKEND}_${BEST}.pkl"
export MUJOCO_SCENE="$SCENE/mujoco_scene"
```

`BACKEND` 和 `BEST` 只是示例，必须以 benchmark JSON 为准。

## 5. 生成并查看带 mesh 的 MuJoCo

```bash
"$PY" tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --mesh-proxy "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --output-dir "$MUJOCO_SCENE"

"$PY" tools/l25_plan_to_trajectory.py \
  --plan "$PLAN" --output "$TRAJECTORY" --frames 60

"$PY" tools/l25_mujoco_playback.py \
  --model "$MUJOCO_SCENE/l25_object_relative_scene.xml" \
  --trajectory "$TRAJECTORY" --fps 30 --no-loop
```

观察手指是否穿模、物体是否错位、抓取姿态是否合理。MuJoCo 不包含机械臂。

## 6. L25 真机：先离线，再只读，再有界动作
## 6. L25 真机：先离线，再只读，再有界动作

离线检查：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" --frame 0 \
  --report "$SCENE/l25_offline.json"
```

检查 CAN，不由工具自动配置：

```bash
ip link show "$CAN_IFACE"
```

只读读取 25 路状态：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" --frame 0 \
  --sdk-package "$L25_SDK" --can "$CAN_IFACE" \
  --read-state --report "$SCENE/l25_preflight.json"
```

第一次真实动作只允许少量通道：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" --frame 0 \
  --sdk-package "$L25_SDK" --can "$CAN_IFACE" \
  --read-state --hardware \
  --confirm L25_RIGHT_CLEAR \
  --channels 0,1,2 --max-step 1 \
  --speed 20 --torque 20 \
  --report "$SCENE/l25_channel_test.json"
```

确认方向和限位后，才可以将全部通道缓慢移动到一个目标帧：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" --frame 0 \
  --sdk-package "$L25_SDK" --can "$CAN_IFACE" \
  --read-state --hardware \
  --confirm L25_RIGHT_CLEAR --replay-target \
  --rate-hz 5 --max-step 1 \
  --speed 20 --torque 20 \
  --report "$SCENE/l25_target_test.json"
```

当前命令只执行一个 `--frame`，不执行完整 60 帧轨迹；完整连续真机抓取执行器尚未提供。
