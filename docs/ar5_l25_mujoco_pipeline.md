# AR5 + L25 联合 MuJoCo 仿真

本文说明机械臂和灵巧手的离线联合仿真：

```text
L25 抓取轨迹 + AR5 关节轨迹
  -> AR5 FK 得到末端位姿
  -> AR5 末端挂接 L25
  -> MuJoCo 执行联合运动
  -> RGB / Depth / 触觉 / 抬升成功标签
```

这是离线 MuJoCo 流程，不连接真实 AR5、L25、CAN 或 Luban ROS。

## 0. 设置路径

```bash
export RETARGET_ROOT=/path/to/AnyDexRetarget
export PY="$RETARGET_ROOT/.venv/bin/python"
export SCENE=/path/to/l25_object_relative_scene.xml
export L25_TRAJECTORY=/path/to/l25_trajectory.pkl
export ARM_TRAJECTORY=/path/to/ar5_joint_trajectory.npz
export AR5_URDF=/path/to/AR5-5_08R-W4C4A6-ZY2.urdf
export FLANGE_HAND=/path/to/T_arm_flange_l25_hand.npy
export OUTPUT=/path/to/outputs/ar5_l25_episode.npz
export RENDER_DIR=/path/to/outputs/ar5_l25_render

cd "$RETARGET_ROOT"
"$PY" -c "import mujoco; print('MuJoCo', mujoco.__version__)"
```

## 1. 输入文件要求

### L25 轨迹

`$L25_TRAJECTORY` 是 pickle 列表，每一帧包含 21 维弧度制 `target`：

```text
[{"target": [q0, ..., q20]}, ...]
```

如果从物体相对计划生成：

```bash
"$PY" tools/l25_plan_to_trajectory.py \
  --plan /path/to/l25_collision_aware_plan.npz \
  --output "$L25_TRAJECTORY" --frames 60
```

### AR5 关节轨迹

`$ARM_TRAJECTORY` 是 NPZ，至少包含：

```text
joint_names: (7,)
positions: (N, 7)
```

可选 `timestamps: (N,)`。关节名应能被重排为 `r_joint_1` 到 `r_joint_7`。

如果原始轨迹顺序来自 Luban，可以先转换并检查：

```bash
"$PY" tools/convert_ar5_trajectory_fk.py \
  --trajectory "$ARM_TRAJECTORY" \
  --urdf "$AR5_URDF" \
  --output "$OUTPUT.arm_fk.npz"
```

这里的转换输出主要用于检查 AR5 末端位姿；联合 episode 工具会在内部执行同样的 FK。

### 机械臂末端到 L25 手基座标定

`$FLANGE_HAND` 是 4x4 的 `T_arm_flange_l25_hand.npy`。没有这份标定时可以省略参数，但手会使用单位安装变换，不能用于真实安装关系验证。

## 2. 检查联合场景

联合 scene 必须包含：

```text
hand_base_link
reconstructed_object
L25 的 21 个关节
```

如果从 L25 物体相对 plan 生成场景：

```bash
"$PY" tools/build_l25_object_relative_scene.py \
  --plan /path/to/l25_collision_aware_plan.npz \
  --mesh-proxy /path/to/object_mesh_anchor.ply \
  --output-dir /path/to/mujoco_scene
```

然后：

```bash
export SCENE=/path/to/mujoco_scene/l25_object_relative_scene.xml
```

该基础 scene 是 L25 + 物体模型；联合执行器会把 L25 手基座作为 AR5 末端的子级运动目标。

## 3. 执行 AR5 + L25 联合 episode

```bash
"$PY" tools/run_ar5_l25_mujoco_episode.py \
  --scene "$SCENE" \
  --l25-trajectory "$L25_TRAJECTORY" \
  --arm-trajectory "$ARM_TRAJECTORY" \
  --urdf "$AR5_URDF" \
  --flange-hand "$FLANGE_HAND" \
  --output "$OUTPUT" \
  --fps 20 \
  --contact-force-threshold 0.1 \
  --lift-m 0.05 \
  --kinematic-hold \
  --render-dir "$RENDER_DIR" \
  --render-width 640 \
  --render-height 480
```

参数说明：

- `--arm-trajectory`：AR5 7 关节轨迹；
- `--urdf`：用于 FK 的 AR5 URDF；
- `--flange-hand`：AR5 法兰到 L25 手基座的 4x4 变换；
- `--lift-m 0.05`：后半段抬升 5 cm；
- `--kinematic-hold`：检测到至少两指接触后，保持物体相对手的位姿；
- `--render-dir`：保存 MuJoCo RGB 和 Depth 数组。

输出：

```text
$OUTPUT
$OUTPUT.json
$RENDER_DIR/rgb.npy
$RENDER_DIR/depth.npy
```

`$OUTPUT` 中包含时间戳、L25 qpos、五指接触、wrench、手位置和物体位置。JSON 中包含接触数量、是否检测到抓取、抬升高度和 `lift_success`。

## 4. 检查结果

```bash
cat "$OUTPUT.json"
RENDER_DIR="$RENDER_DIR" "$PY" -c '
import os
from pathlib import Path
import numpy as np
render = Path(os.environ["RENDER_DIR"])
rgb = np.load(render / "rgb.npy")
depth = np.load(render / "depth.npy")
print("rgb:", rgb.shape, rgb.dtype)
print("depth:", depth.shape, depth.dtype)
' 
```

重点检查：

- AR5 末端是否沿预期轨迹运动；
- L25 手基座是否跟随 AR5 法兰；
- 手指与物体是否穿模；
- 至少两指接触是否出现；
- 抬升后物体位置是否随手上升；
- `lift_success` 是否只是仿真运动学判据。

`kinematic_hold` 是运动学保持，不是动力学力闭合证明；成功标签不能直接推断真机一定成功。

## 5. 只做不抬升的联合检查

```bash
"$PY" tools/run_ar5_l25_mujoco_episode.py \
  --scene "$SCENE" \
  --l25-trajectory "$L25_TRAJECTORY" \
  --arm-trajectory "$ARM_TRAJECTORY" \
  --urdf "$AR5_URDF" \
  --flange-hand "$FLANGE_HAND" \
  --output "$OUTPUT.no_lift.npz" \
  --fps 20 \
  --render-dir "$RENDER_DIR/no_lift"
```

## 6. 与 Luban MuJoCo 的关系

本手册的 `run_ar5_l25_mujoco_episode.py` 是 AnyDexRetarget 内的离线验证器，不需要启动 Luban。Luban bringup 是另一条 ROS 2 控制器集成检查路径，依赖 Luban 工作区、`mujoco_hardware` 和相应 planner；它不是本 episode 记录器的前置条件。

## 7. 当前边界

- AR5 轨迹必须由外部规划器、示教或已有 NPZ 提供；本工具不自动求 AR5 碰撞规划；
- 本流程不发送 ROS topic，不控制真实机械臂或真实灵巧手；
- 触觉来自 MuJoCo 接触几何；
- `--kinematic-hold` 不等于真实物理抓取；
- 真机执行请使用 `docs/l25_real_hardware_runbook.md`，且当前仅支持 L25 单目标有界动作。
