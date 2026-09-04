# AR5 + LinkerHand L25 真机抓取实施手册

本文面向在新机器上部署完整的机械臂抓取链路。目标是让珞石 AR5 携带 LinkerHand L25，依据 RGB-D/3DGeneration/HUG 生成的抓取结果完成：

```text
RGB-D / 3DGeneration
  -> HUG 生成 50 个候选
  -> L25 重定向、物体相对优化和碰撞排序
  -> 选定一个候选
  -> 结合目标机器标定生成 AR5 法兰目标
  -> Luban 规划 pregrasp / approach / lift
  -> L25 preshape / close
  -> AR5 抬升
```

本文不包含触觉，也不使用 MuJoCo 作为真机控制器。MuJoCo 只能作为真机前的强制检查环节。

## 1. 已验证能力和边界

部署分支 `feature/robot-deployment-zed2i-gemini-l25` 已提供：

- HUG 50 候选生成；
- L25 多后端重定向和物体相对碰撞排序；
- 抓取合同导出；
- 相机锚点到 AR5 基座的坐标变换；
- AR5 法兰目标和 L25 16 主动关节目标生成；
- Luban ROS2 分阶段真机执行器；
- `preview / preshape / pregrasp / approach / close / attach / lift` 阶段。

真机执行器故意要求每个阶段单独调用。真实模式禁止 `--stage all`，操作员必须观察上一阶段状态后再执行下一阶段。

当前不提供：

- 无人值守的一键抓取；
- 真机触觉闭环；
- 自动 release 阶段；
- 对任意物体的成功保证。

## 2. 新机器和分支

```bash
git clone --branch feature/robot-deployment-zed2i-gemini-l25 \
  <仓库地址> /path/to/AnyDexRetarget
cd /path/to/AnyDexRetarget
```

重新创建 Python 环境，不要复制旧机器的 `.venv`：

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install mujoco
```

设置路径：

```bash
export RETARGET_ROOT=/path/to/AnyDexRetarget
export PY="$RETARGET_ROOT/.venv/bin/python"
export SCENE="$RETARGET_ROOT/outputs/p5636_motor/run_001"
export GEN_MESH=/path/to/3dgeneration/p5636_motor.glb
export T_ANCHOR_OBJECT=/path/to/calibration/T_anchor_object.json
export BASE_ANCHOR=/path/to/calibration/T_robot_base_anchor_capture.npz
export FLANGE_HAND=/path/to/calibration/T_arm_flange_l25_hand.npz
export LUBAN_ROOT=/path/to/luban_framework

cd "$RETARGET_ROOT"
```

所有 `/path/to/...` 都必须替换成目标机器的真实路径。

## 3. 输入和标定要求

### RGB-D 输入

```text
$SCENE/view_000/rgb.png
$SCENE/view_000/depth.png
$SCENE/view_000/intrinsics.txt
$SCENE/view_000/object_pointcloud.npz
```

### 3DGeneration mesh

`$GEN_MESH` 必须是实际导出的物体 mesh。mesh 的单位和位姿必须明确，不能使用 Blender 世界原点下未经对齐的模型。

### 标定矩阵

`$T_ANCHOR_OBJECT`：物体坐标系到相机 anchor frame 的 4x4 变换。

`$BASE_ANCHOR`：NPZ 中包含 `T_robot_base_anchor_capture`，将相机采集时的 anchor frame 变换到 AR5 `r_base_link`。

`$FLANGE_HAND`：NPZ 中包含 `T_arm_flange_l25_hand`，表示 AR5 法兰到 L25 手基座的安装变换。

这些矩阵必须在目标机器、目标安装姿态下重新确认，不能直接照搬另一台机器。

## 4. 生成 50 个 HUG 候选

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
  --robot l25 \
  --optimizer vector \
  --candidates 50 \
  --sampling-steps 50 \
  --frames 60 \
  --fps 30 \
  --output "$HUG_OUTPUT" \
  --dry-run
```

每个候选会生成 `prediction.pkl`、`canonical_grasp.npz`、`trajectory.pkl`、`trajectory.npz` 和 `metrics.json`。`trajectory.pkl` 是静态 L25 21 维 qpos 目标的重复帧，不是机械臂轨迹。

## 5. L25 优化和候选选择

```bash
export BENCHMARK_OUTPUT="$SCENE/backend_benchmark_50"

"$PY" tools/benchmark_l25_retarget_backends.py \
  --candidates-dir "$HUG_OUTPUT" \
  --object-mesh "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --output-dir "$BENCHMARK_OUTPUT"

cat "$BENCHMARK_OUTPUT/backend_benchmark.json"
```

最终选择依据包括指尖到物体的误差、拇指对向、物体穿透、自碰撞、关节限位余量和姿态偏移。设置实际选中的候选：

```bash
export BACKEND=vector
export BEST=candidate_017
export PLAN="$BENCHMARK_OUTPUT/$BACKEND/$BEST/l25_collision_aware_plan.npz"
```

`candidate_017` 只是示例，必须根据 `backend_benchmark.json` 修改。

## 6. MuJoCo 强制检查

在真机前先生成带物体 mesh 的 L25 场景并查看：

```bash
export MUJOCO_SCENE="$SCENE/mujoco_scene"
export L25_TRAJECTORY="$SCENE/l25_selected.pkl"

"$PY" tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --mesh-proxy "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --output-dir "$MUJOCO_SCENE"

"$PY" tools/l25_plan_to_trajectory.py \
  --plan "$PLAN" --output "$L25_TRAJECTORY" --frames 60

"$PY" tools/l25_mujoco_playback.py \
  --model "$MUJOCO_SCENE/l25_object_relative_scene.xml" \
  --trajectory "$L25_TRAJECTORY" --fps 30 --no-loop
```

如果 MuJoCo 中 mesh 错位、穿模或手型不合理，停止，不进入真机。

## 7. 导出抓取合同

```bash
export CONTRACT="$SCENE/grasp_execution_contract.npz"

"$PY" tools/export_grasp_execution_plan.py \
  --plan "$PLAN" \
  --object-mesh "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
  --reconstruction-result \
    "$SCENE/3dgeneration/reconstruction/reconstruction_metadata.json" \
  --anchor-frame hand_camera_color_optical_frame \
  --hand-side right \
  --candidate-id "$BEST" \
  --output "$CONTRACT"
```

该命令只生成机器人无关的抓取合同，不发送 ROS 或硬件命令。

## 8. 计算 AR5 pregrasp、抓取和抬升目标

`--pregrasp-offset-hand-m` 是目标机器专用参数，表示从抓取姿态沿 L25 手部局部坐标系退开的距离。先用低风险姿态测量并填写，不能照抄示例。

```bash
export REQUEST="$SCENE/luban_grasp_request.npz"

"$PY" tools/prepare_luban_grasp_request.py \
  --grasp-contract "$CONTRACT" \
  --base-anchor-capture "$BASE_ANCHOR" \
  --flange-hand "$FLANGE_HAND" \
  --base-frame r_base_link \
  --anchor-frame hand_camera_color_optical_frame \
  --pregrasp-offset-hand-m 0.00 0.00 -0.08 \
  --output "$REQUEST"
```

输出请求包含：

```text
T_robot_base_arm_flange_pregrasp
T_robot_base_arm_flange_target
T_robot_base_arm_flange_lift
l25_preshape_positions
l25_active_positions
```

## 9. 启动 Luban 真机节点

在第二个终端进入 Luban 工作区。真机启动不能设置 `MUJOCO=1` 或 `MOCK=1`：

```bash
cd "$LUBAN_ROOT"
source /opt/ros/jazzy/setup.bash
source .ws/devel/setup.bash

unset MUJOCO
unset MOCK
export RVIZ=1
export LUBAN_ROBOT_MODEL=dual_arm_ar508_l20
export RIGHT_HAND_MODEL=l25
export LEFT_HAND_MODEL=l25
export ARM_SIDE=right
export LUBAN_MOTION_PLANNER=ompl

ros2 launch luban_bringup launch_onboard_nodes.py target:=real
```

如果目标机器已安装并验证 cuMotion，可将 planner 改为：

```bash
export LUBAN_MOTION_PLANNER=cumotion
```

启动日志必须显示真实硬件插件、右臂和右手控制器已激活。不要把 `target:=simulation`、`MUJOCO=1` 或 `MOCK=1` 的启动结果当成真机就绪。

另开终端检查：

```bash
source /opt/ros/jazzy/setup.bash
source "$LUBAN_ROOT/.ws/devel/setup.bash"
ros2 control list_controllers
ros2 topic list | rg 'right_arm|right_hand|joint_states'
```

## 10. 真机分阶段执行

先只预览，不发送动作：

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" \
  --stage preview \
  --runtime real
```

确认打印的 `pregrasp_position_m`、`grasp_position_m` 和 `approach_distance_m` 正确后，按以下顺序执行。每个阶段完成后都要人工检查机器人状态。

### 10.1 L25 预抓取姿态

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage preshape --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

### 10.2 AR5 移动到 pregrasp

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage pregrasp --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

这一步会启用目标物体的规划障碍并等待 AR5 完成信号。

### 10.3 AR5 接近物体

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage approach --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

默认限制 pregrasp 到抓取位姿的平移距离不超过 0.12 m。

### 10.4 L25 闭合

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage close --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

这一步发送 16 个 L25 主动关节目标，Luban 的手部控制器负责下发到硬件。

### 10.5 附着碰撞模型

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage attach --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

该阶段只向规划系统发布附着碰撞物体，不额外驱动手。

### 10.6 AR5 小幅抬升

```bash
"$PY" tools/luban_ros_grasp_execute.py \
  --request "$REQUEST" --stage lift --runtime real \
  --execute --confirm AR5_L25_CLEAR
```

当前实现限制抬升距离不超过 0.05 m。抬升前确认物体没有被桌面、线缆或周边结构卡住。

## 11. 禁止事项和回退

- 真机不要使用 `--stage all`；程序会拒绝该模式；
- 真机不要同时运行 `l25_hardware_execute.py` 和 Luban 手部控制器，避免两个进程同时占用 L25 CAN；
- 不要在未完成 MuJoCo 检查、标定确认或急停检查时执行 `--execute`；
- 任意阶段超时、控制器未激活、目标偏离、手指方向反转时，立即停止并断开执行流程；
- 当前没有 release 阶段，抬升后应由操作员手动规划安全放置流程。

## 12. 每次实验保存的文件

至少保留：

```text
grasp_execution_contract.npz
grasp_execution_contract.json
luban_grasp_request.npz
luban_grasp_request.json
MuJoCo 场景和回放轨迹
每个阶段的终端日志
目标机器标定矩阵版本
Luban 与 AnyDexRetarget commit
```

该记录用于复现和故障定位。仿真成功或规划器返回成功不等于真实物体一定抓取成功。
