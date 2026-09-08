# LinkerHand L25 真机运行手册

本文说明从离线轨迹到 LinkerHand L25 真机的当前可执行流程。范围仅包含右手 L25，不包含 AR5、Luban ROS 或机械臂规划。

## 当前能力边界

当前仓库已经具备：

- L25 qpos（21 个关节）到 LinkerHand SDK 0..255 通道的映射；
- 轨迹文件格式、关节限位和 NaN 检查；
- MuJoCo 中带物体 mesh 的 L25 场景回放；
- CAN 接口和 SDK 的只读状态预检；
- 单个目标帧的有界真机动作；
- 将当前状态缓慢移动到一个目标帧的 `--replay-target`。

当前尚未具备：

- 60 帧轨迹的连续真机执行器；
- AR5 与 L25 的同步真机执行；
- 真机接触力闭环或抓取成功保证。

因此，当前真机流程用于空中标定和单目标验证。不得把 `--replay-target` 当作完整抓取轨迹执行。

## 1. 新机器准备

在新机器上克隆包含本工具的分支：

```bash
git clone --branch feature/robot-deployment-zed2i-gemini-l25 \
  <仓库地址> /path/to/AnyDexRetarget
cd /path/to/AnyDexRetarget
```

不要复制其他机器的 `.venv`。重新创建环境：

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install mujoco
```

设置本机路径。只有这一段需要按机器修改：

```bash
export RETARGET_ROOT=/path/to/AnyDexRetarget
export PY="$RETARGET_ROOT/.venv/bin/python"
export SCENE="$RETARGET_ROOT/outputs/l25_session_001"
export L25_SDK=/path/to/linkerhand_sdk
export CAN_IFACE=can0

cd "$RETARGET_ROOT"
```

SDK 目录必须包含：

```text
$L25_SDK/LinkerHand/linker_hand_api.py
```

检查 Python 和 MuJoCo：

```bash
"$PY" --version
"$PY" -c "import mujoco; print(mujoco.__version__)"
```

## 2. 准备 L25 轨迹

`l25_hardware_execute.py` 接受一个 pickle 文件，格式为非空列表，每一项包含一个 21 维弧度制 `target`：

```text
[
  {"target": [q0, q1, ..., q20]},
  ...
]
```

如果已有物体相对优化计划：

```bash
export PLAN=/path/to/l25_collision_aware_plan.npz
export TRAJECTORY="$SCENE/l25_candidate.pkl"

"$PY" tools/l25_plan_to_trajectory.py \
  --plan "$PLAN" \
  --output "$TRAJECTORY" \
  --frames 60
```

如果已有 HUG 预测，也可以直接重定向：

```bash
"$PY" tools/hug_static_retarget.py \
  --prediction /path/to/grasp_prediction.pkl \
  --optimizer vector \
  --output "$TRAJECTORY" \
  --frames 60
```

## 3. 先在 MuJoCo 查看相同目标

如果已有带物体 mesh 的计划，生成场景：

```bash
export MESH=/path/to/object_mesh_anchor.ply
export MUJOCO_SCENE="$SCENE/mujoco_scene"

"$PY" tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --mesh-proxy "$MESH" \
  --output-dir "$MUJOCO_SCENE"
```

必须显式传 `--mesh-proxy`。这样计划中记录的旧机器绝对路径不会被使用。

查看手、物体和轨迹：

```bash
"$PY" tools/l25_mujoco_playback.py \
  --model "$MUJOCO_SCENE/l25_object_relative_scene.xml" \
  --trajectory "$TRAJECTORY" \
  --fps 30 \
  --no-loop
```

如果 MuJoCo 中已经穿模、物体位置不对或手型明显错误，禁止继续真机。

## 4. 离线指令检查

这一条不打开 SDK、不访问 CAN，也不会运动：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" \
  --frame 0 \
  --report "$SCENE/l25_offline_frame0.json"
```

确认输出中的：

- `qpos_finite=true`；
- `target_command_0_255` 全部在 0..255；
- 目标帧是 MuJoCo 中实际检查过的帧。

## 5. 配置并检查 CAN

本工具不会自动修改 CAN 配置。由现场人员按照 L25 SDK 和 CAN 适配器要求配置接口，然后检查：

```bash
ip link show "$CAN_IFACE"
```

输出必须显示接口为 `state UP`。接口不是 `can0` 时修改 `CAN_IFACE`。

## 6. SDK 只读预检

这一步只读取当前 25 通道状态，不发送运动命令：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" \
  --frame 0 \
  --sdk-package "$L25_SDK" \
  --can "$CAN_IFACE" \
  --read-state \
  --report "$SCENE/l25_preflight.json"
```

只有以下条件全部满足，才能进入动作测试：

- SDK 文件路径正确；
- CAN 接口为 UP；
- SDK 返回 25 个有限的 0..255 状态值；
- `state_target_max_delta` 与当前测试目标相符；
- 手周围没有物体、人员或线缆干涉；
- 急停或断电手段可立即使用。

## 7. 单通道有界动作

第一次只允许少数通道，每次调用最多发送一个小步长：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" \
  --frame 0 \
  --sdk-package "$L25_SDK" \
  --can "$CAN_IFACE" \
  --read-state \
  --hardware \
  --confirm L25_RIGHT_CLEAR \
  --channels 0,1,2 \
  --max-step 1 \
  --speed 20 \
  --torque 20 \
  --report "$SCENE/l25_channel_test.json"
```

`--channels` 是 SDK 通道号，不是 21 维 qpos 下标。应先用厂商 SDK 文档确认通道含义。

## 8. 单目标帧缓慢回放

确认单通道动作方向、限位和停止手段正确后，才允许将全部通道缓慢移动到一个目标帧：

```bash
"$PY" tools/l25_hardware_execute.py \
  --trajectory "$TRAJECTORY" \
  --frame 0 \
  --sdk-package "$L25_SDK" \
  --can "$CAN_IFACE" \
  --read-state \
  --hardware \
  --confirm L25_RIGHT_CLEAR \
  --replay-target \
  --rate-hz 5 \
  --max-step 1 \
  --speed 20 \
  --torque 20 \
  --report "$SCENE/l25_target_frame0.json"
```

此命令只执行 `--frame 0`，不会自动执行 `frame 1` 到最后一帧。

## 9. 目前不能做的事情

以下命令不存在，也不应伪造为已支持：

```text
自动执行 60 帧 L25 真机轨迹
L25 触觉闭环调力
AR5 + L25 同步抓取
根据 MuJoCo 成功标签直接保证真机成功
```

要实现完整真机抓取，还需要单独开发并验证一个连续轨迹执行器：逐帧读取目标、限速发送、读取反馈、异常立即停止，并记录每帧的目标和实测状态。在该执行器完成前，真机只按本手册进行单目标、有界、空载验证。

## 10. 报告和故障排查

每次测试保留 `--report` JSON。至少记录：

- Git commit 和分支；
- 轨迹文件路径及候选编号；
- CAN 接口名；
- 目标 0..255 指令；
- 测试前、测试后状态；
- 速度、力矩、最大步长和允许通道。

常见错误：

```text
L25 vendor SDK not found
```

检查 `$L25_SDK/LinkerHand/linker_hand_api.py`。

```text
can0 is not UP
```

检查 CAN 适配器、接口名和现场 CAN 配置；工具不会替你启动 CAN。

```text
real motion requires --confirm L25_RIGHT_CLEAR
```

这是故意的安全闸门。确认现场清空并具备急停后，再显式输入确认字符串。
