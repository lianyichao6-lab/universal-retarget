# ROS 感知 → AnyDex 抓取 全流程 CLI(evo-station-c01 已验证)

状态日期:2026-09-04

本机(`evo-station-c01`)实测跑通:把在线 Luban 感知(ZED2i + RobotPerception)
的输出桥接进 AnyDex,不经过 Hunyuan 重建,直接 HUG → L25 → 真机执行 plan。

## 前置

- **感知在线**:Luban 感知(`luban_framework/src/perception`),发布
  `/clawbot_cam_head/...`(相机图 / 双目深度 / mask / 物体点云)与 `/botclaw/...`。
- **AnyDex 环境** `.venv`:torch 2.8.0+cu128 + HUG + mujoco + manotorch + chumpy
  + dex-retargeting,全部可导入(`import torch, hug, anydexretarget, mujoco, ...`)。
- **桥接 mesh 用的环境**:`~/miniconda3/envs/robot_perception`(Python 3.12,
  同时有 `rclpy` 与 `trimesh`)。先 `source /opt/ros/jazzy/setup.bash`。

## 换物体快速清单

换目标物体只需改 **3 个变量**,其余命令原样照跑:

| 变量 | 含义 | 例子(盒子) | 例子(P5636_36A) |
|---|---|---|---|
| `TAG` | 感知 mesh_cloud 的物体 tag | `hezi` | `P5636_36A__20250920001` |
| `SCENE` | 输出目录 | `run_001` | `p5636_36a` |
| `BEST` | 最终候选号(读 benchmark 结果定) | `candidate_019` | `candidate_013` |

命令里统一用 `$TAG` / `$SCENE` / `$BEST` 占位:

```bash
TAG=hezi                 # 换成新物体的 tag
SCENE=run_001            # 换成新场景目录名
# BEST 等第 ⑤ 步读 benchmark 结果后再定
```

顺序固定(桥接目录 ① 必须先于 ②,都写 `outputs/perception_bridge/$SCENE/`):

```bash
# ① 桥接可见面(先!)
/usr/bin/python3 tools/capture_ros_perception.py --output outputs/perception_bridge/$SCENE

# ② 桥接 mesh(后!)
~/miniconda3/envs/robot_perception/bin/python tools/capture_ros_object_mesh.py \
  --cloud-topic /clawbot_cam_head/perception/mesh_cloud/obj_$TAG \
  --output outputs/perception_bridge/$SCENE/object_mesh_camera.ply
```

其余 ③~⑥ 及真机执行全用 `outputs/perception_bridge/$SCENE/...` 和 `$BEST`,见下文各节。

## 流程总览

```text
Luban 感知(RGB-D + 分割 + 6D 位姿)
  -> ① 桥接可见面点云(object_pointcloud.npz)
  -> ② 桥接物体 mesh(object_mesh_camera.ply,world->camera)
  -> ③ HUG 50 候选
  -> ④ 四后端 benchmark(collision-aware)
  -> ⑤ 选 best
  -> ⑥ trajectory + 执行 plan(TCP 位姿 + 手势 qpos)
```

---

## ① 桥接可见面点云(ROS → AnyDex 场景文件)

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 tools/capture_ros_perception.py \
  --output outputs/perception_bridge/run_001 --keyboard
```

摆好物体后按回车保存,`q` 退出。产物:

```text
rgb.png                    left_color/image_raw
depth.png                  perception/stereo_depth(uint16 mm)
intrinsics.txt             left_color/camera_info 的 K
mask.png / mask.json       perception/masks + 前景点
object_pointcloud.npz      可见面(相机系,points_camera/pixels_uv)
capture_metadata.json      元数据
```

## ② 桥接物体 mesh(CAD 点云 → 相机系,给碰撞检测)

```bash
source /opt/ros/jazzy/setup.bash
~/miniconda3/envs/robot_perception/bin/python tools/capture_ros_object_mesh.py \
  --cloud-topic /clawbot_cam_head/perception/mesh_cloud/obj_hezi \
  --output outputs/perception_bridge/run_001/object_mesh_camera.ply
```

订阅 `mesh_cloud/obj_<tag>`(world 系),用 `T_world_headcam` 转到相机系,凸包重建。
产物 `object_mesh_camera.ply`(代替 Hunyuan 重建 mesh,当 `--object-mesh` 用)。

> 坐标系验证:mesh 与可见面点云在相机系几乎重合(mesh bounds 略大于可见面,
> 因为 mesh 含背面)。

## ③ HUG 50 候选

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python tools/generate_hug_candidates.py \
  --rgb outputs/perception_bridge/run_001/rgb.png \
  --depth outputs/perception_bridge/run_001/depth.png \
  --intrinsics outputs/perception_bridge/run_001/intrinsics.txt \
  --pointcloud outputs/perception_bridge/run_001/object_pointcloud.npz \
  --robot l25 --optimizer vector --candidates 50 --dry-run \
  --output outputs/perception_bridge/run_001/hug_candidates_50
```

产物 `hug_candidates_50/`(50 候选)+ `candidates.csv` + `best_candidate.json`。

## ④ 四后端 benchmark(collision-aware)

```bash
.venv/bin/python tools/benchmark_l25_retarget_backends.py \
  --candidates-dir outputs/perception_bridge/run_001/hug_candidates_50 \
  --object-mesh outputs/perception_bridge/run_001/object_mesh_camera.ply \
  --output-dir outputs/perception_bridge/run_001/backend_benchmark_50
```

产物:
- `_shared/object_collision_proxy.ply`(碰撞代理)+ `_shared/contact_plans/`(接触计划)
- 每后端目录下的 `best_l25_candidates.json` 与每个候选的 `l25_collision_aware_plan.npz`

> 只要跑某几个后端,加 `--backends vector adaptive` 跳过其余。共享的碰撞代理与
> 接触计划只算一次,后面端打印 `contact cache hit`。

## ⑤ 选 best

```bash
.venv/bin/python -c "
import json
d = json.load(open('outputs/perception_bridge/run_001/backend_benchmark_50/vector/best_l25_candidates.json'))
for c in d['top_candidates']:
    if c['recommended']:
        print(c['candidate'], 'score=%.2f' % c['final_l25_score'])
"
```

本例选 `candidate_019`(vector 唯一过全部门限:5 指接触、穿透 0.01mm、关节余量 0.061)。

## ⑥ trajectory + 执行 plan

```bash
BEST=candidate_019
PLAN=outputs/perception_bridge/run_001/backend_benchmark_50/vector/$BEST/l25_collision_aware_plan.npz

.venv/bin/python tools/l25_plan_to_trajectory.py --plan "$PLAN" \
  --output outputs/perception_bridge/run_001/l25_${BEST}_trajectory.pkl --frames 60

.venv/bin/python tools/export_grasp_execution_plan.py --plan "$PLAN" \
  --object-mesh outputs/perception_bridge/run_001/object_mesh_camera.ply \
  --anchor-frame zed_left_camera_frame_optical \
  --candidate-id $BEST \
  --output outputs/perception_bridge/run_001/grasp_execution_plan.npz
```

## 可视化(可选)

```bash
# 交互式 MuJoCo(手 + 物体 3D,可旋转缩放;显示在 :0 屏)
DISPLAY=:0 .venv/bin/python tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" --output-dir outputs/perception_bridge/run_001/mujoco_view --show

# 静态渲染图(离屏,GLFW 需 DISPLAY)
DISPLAY=:0 MUJOCO_GL=glfw .venv/bin/python tools/render_l25_scene.py \
  --scene outputs/perception_bridge/run_001/backend_benchmark_50/vector/$BEST/collision_aware_scene/l25_object_relative_scene.xml \
  --output outputs/perception_bridge/run_001/grasp_$BEST.png
```

## 最终输出

| 文件 | 内容 |
|---|---|
| `grasp_execution_plan.npz` | `T_anchor_l25_hand`(TCP 位姿)+ `l25_qpos`(21 关节手势) |
| `l25_candidate_019_trajectory.pkl` | 60 帧轨迹(回放) |

## 真机执行(仅 L25 手,LinkerHand CAN)

### 驱动

LinkerHand SDK(U 盘 `drivers/linkerhand-ros2-sdk`,v3.1.1)拷到 `~/linker_hand_ros2_sdk`。
`--sdk-package` 指向含 `LinkerHand/linker_hand_api.py` 的目录:

```bash
export LINKERHAND_SDK_PACKAGE=~/linker_hand_ros2_sdk/linker_hand_ros2_sdk/linker_hand_ros2_sdk
```

依赖 `python-can`(系统 `/usr/bin/python3` 已有)。CAN 接口需先 UP(`ip link show can0`),
工具不自动配置 CAN。

### 只读预检(不动机器)

```bash
/usr/bin/python3 tools/l25_hardware_execute.py \
  --sdk-package "$LINKERHAND_SDK_PACKAGE" \
  --trajectory outputs/perception_bridge/run_001/l25_candidate_019_trajectory.pkl \
  --frame 0 --read-state \
  --report outputs/perception_bridge/run_001/l25_preflight.json
```

### 真机执行(动 L25 手)

```bash
/usr/bin/python3 tools/l25_hardware_execute.py \
  --sdk-package "$LINKERHAND_SDK_PACKAGE" \
  --trajectory outputs/perception_bridge/run_001/l25_candidate_019_trajectory.pkl \
  --frame 0 \
  --hardware --confirm L25_RIGHT_CLEAR \
  --read-state \
  --channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 \
  --replay-target --rate-hz 10 --max-step 2 \
  --speed 30 --torque 40 \
  --report outputs/perception_bridge/run_001/l25_execute.json
```

关键点:

- `--hardware` **必须同时带 `--read-state`**(先读状态验证反馈再动,否则报错)。
- `--confirm L25_RIGHT_CLEAR` 必须原样。
- `--replay-target` = 慢速有界逼近(每通道每步最多变 `--max-step`,10Hz,上限 60 步)。
- **L25 需使能**:工具已在 `set_speed`/`set_torque` 前调 `api.set_enable()`(SDK 发 0x85
  使能电机);否则手默认失能,`finger_move` 发位置命令也不动(症状:`target max delta`
  多轮不变)。
- 目标距离大(如 state↔target 差 >120)时,60 步上限不够会报
  `replay exceeded the 60-step safety limit` —— **再跑一遍同命令**继续逼近即可。

### 已知坑

- 桥接目录顺序:先 ① 可见面、后 ② mesh(都写同一目录;反了会导致
  `object_pointcloud.npz` 里 `source_mask` 路径失效,需 `--point` 补救)。
- 大物体(对角线 >12cm)可能无候选过全部门限:关节余量健康的候选接触误差超 10mm,
  接触误差达标的候选关节饱和(余量~0),是"手偏小/物体偏大"的矛盾。此时选
  关节健康、接触误差略超的候选(如 P5636_36A 的 `candidate_013`,接触误差 10.1mm)。

## 手臂 + 手(真机待补)

执行公式 `T_robot_base_arm_flange = T_robot_base_anchor @ T_anchor_l25_hand @ inv(T_arm_flange_l25_hand)`。

计划中 `required_target_transforms` = `T_robot_base_anchor`、`T_arm_flange_l25_hand`
+ pregrasp 偏移(`pregrasp_defined: False`)。补上后走 Luban 分阶段执行
(`preview` → `pregrasp` → `approach` → `close`,每步确认)。
