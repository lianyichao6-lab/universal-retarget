中文 | [English](README.md)

# Universal Retarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#当前状态)

Universal Retarget 是一个面向异构灵巧手的抓取生成与跨本体动作重定向研究平台，目标是把“人手层面的抓取意图”转换成具体机械手可执行的姿态。当前已验证链路包括 RGB-D 感知、学习式三维补全、HUG MANO 抓取生成、机器人无关中间态、多种 Retargeting 后端、L25 物体相对优化、碰撞修正、MuJoCo 检查以及 LinkerHand 真机有界执行。

项目基于 [AnyDexRetarget](https://github.com/qqsq12321/AnyDexRetarget) 扩展，完整上游历史保存在 [upstream-history](https://github.com/lianyichao6-lab/universal-retarget/tree/upstream-history) 分支。

## 已验证主链路

~~~text
Orbbec Gemini 335 多视角 RGB-D
  -> 交互分割与真实可见面点云
  -> Hunyuan3D-2mv 三维补全并对齐照片
  -> 真实测量 + 生成补全面 Hybrid 点云
  -> HUG 生成 50 个随机 MANO 抓取候选
  -> CanonicalGraspState
  -> AnyDex Vector | AnyDex Adaptive | DexPilot | JointAngle
  -> L25 物体相对 qpos 优化
  -> 物体碰撞修正与 L25 最终排序
  -> MuJoCo 检查
  -> LinkerHand 0..255 CAN 有界执行
~~~

杯子实验已经使用 **Vector + candidate_017** 完成端到端验证，并在真机上成功抓起实物。这是一次实验结果，不代表对任意物体都能保证抓取成功。

## 本仓库新增内容

- **RGB-D 与物体几何**：Gemini 335 交互采集、物体 mask、真实点云、Hunyuan 多视角补全、照片坐标对齐和 Hybrid 点云。
- **HUG 接入**：一次加载模型后批量采样 MANO 抓法，保存候选结果、网页可视化并进行初步排序。
- **统一中间态**：机器人无关的 CanonicalGraspState，保存腕部局部 21 x 3 关键点、MANO pose/shape、778 点 mesh、坐标变换、置信度和物体相对信息。
- **四种 L25 Retargeting 后端**：AnyDex Vector、AnyDex Adaptive、dex-retargeting DexPilot 和 dex-retargeting JointAngle。
- **物体相对规划**：提取 HUG 表面锚点、刚体手物对齐、连续优化 L25 关节、修正物体穿透并重新排序。
- **仿真与真机输出**：L25 MuJoCo 场景、静态 qpos 轨迹、完整 0..255 SDK 映射、只读预检和有界 CAN 回放。
- **机器人模型与标定**：加入 L25、L6、O6 模型资源及 L25 几何标定配置。目前完整自主抓取链路只在右手 L25 上验证。

## 表示层边界

~~~text
HUG MANO grasp
  -> 相机坐标系人手关键点 (21, 3)
  -> CanonicalGraspState
       - 腕部局部关键点 (21, 3)
       - MANO pose / shape
       - MANO mesh (778, 3)
       - 相机与 canonical 坐标变换
       - 物体点与接触元数据
  -> 各 Retargeting 后端的人手目标
  -> L25 弧度制 qpos
  -> LinkerHand 0..255 通道指令
~~~

**21 x 3** 表示 21 个人手关键点，每个点具有 XYZ 三维坐标；它不是 21 自由度关节命令。机器人 qpos 是另一套与具体机械结构相关的关节空间表示。

## 当前状态

| 模块 | 状态 |
|---|---|
| Gemini 335 配准 RGB-D 采集 | 已验证 |
| Hunyuan 多视角 mesh 补全 | 实验性验证 |
| 真实/生成 Hybrid 点云 | 已实现 |
| HUG MANO 多候选生成 | 已验证 |
| CanonicalGraspState | 已实现并进入主链路 |
| Vector / Adaptive / DexPilot / JointAngle | 已在 L25 对比 |
| L25 物体相对优化 | 已实现 |
| L25 与物体穿透修正 | 已实现 |
| MuJoCo L25 + 物体检查 | 已验证 |
| LinkerHand L25 CAN 执行 | 已完成有界回放验证 |
| L6 / O6 完整自主抓取链路 | 尚未验证 |
| VLM 语言目标定位 | 计划中 |
| 机械臂规划和相机到机器人标定 | 不在当前范围 |
| 触觉闭环失败恢复 | 计划中 |

## 安装边界

~~~bash
git clone git@github.com:lianyichao6-lab/universal-retarget.git
cd universal-retarget
~~~

以下大型资源或独立授权组件不会上传到本仓库：

- Python 虚拟环境
- HUG 源码与 checkpoint
- Hunyuan3D 源码与模型权重
- MANO 模型资源
- LinkerHand 厂商 SDK
- RGB-D 采集数据、生成 mesh、预测结果和实验输出

基础包信息位于 [pyproject.toml](pyproject.toml)。复现当前工作站的完整环境还需要上述外部组件；执行前请先阅读 [完整 CLI 文档](docs/cli_reference.md)。

## 从已有场景快速运行

下面从已经具备 anchor RGB-D、真实点云、对齐 Hunyuan mesh 和 Hybrid 点云的场景开始：

~~~bash
SCENE=outputs/reconstruction/object_session_run1

env -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python tools/generate_hug_candidates.py \
  --rgb "$SCENE/view_000/rgb.png" \
  --depth "$SCENE/view_000/depth.png" \
  --intrinsics "$SCENE/view_000/intrinsics.txt" \
  --pointcloud "$SCENE/view_000/object_pointcloud.npz" \
  --hug-pointcloud "$SCENE/hunyuan_hybrid_pointcloud.npz" \
  --robot l25 --optimizer vector \
  --candidates 50 --sampling-steps 50 \
  --frames 60 --fps 30 \
  --output "$SCENE/hug_candidates_50" --dry-run

.venv/bin/python tools/benchmark_l25_retarget_backends.py \
  --candidates-dir "$SCENE/hug_candidates_50" \
  --object-mesh "$SCENE/hunyuan_mv_mesh_photo_aligned.ply" \
  --output-dir "$SCENE/backend_benchmark_50"
~~~

读取 backend_benchmark.json 后再选择后端和候选：

~~~bash
BACKEND=vector
BEST=candidate_017
PLAN="$SCENE/backend_benchmark_50/$BACKEND/$BEST/l25_collision_aware_plan.npz"

.venv/bin/python tools/build_l25_object_relative_scene.py \
  --plan "$PLAN" \
  --output-dir "$SCENE/backend_benchmark_50/$BACKEND/$BEST/mujoco_scene" \
  --show
~~~

从相机启动到真机执行的完整命令见 [docs/cli_reference.md](docs/cli_reference.md)。真机命令要求现场清空、LinkerHand SDK 已安装、CAN 接口已启动，并必须提供执行器要求的明确确认字符串。

## 候选排序依据

最终选择依据 L25 的结果，而不是只看 HUG 初步分数：

- L25 远端指腹代理点到目标位置的平均和最大误差
- 拇指与其他手指的对向关系，以及接触锚点是否分布在物体不同侧
- 物体穿透与不同手指之间的自碰撞
- 关节限位余量与饱和数量
- 相对初始 Retargeting 手型的偏移
- 提供真机报告时的 0～255 指令跟踪误差

Viser 网页中的初步前三名适合排除明显不合理的 MANO 抓法，但不能替代最终面向 L25 的二次排序。

四后端 benchmark 现在共享一份 MuJoCo 碰撞代理 mesh 和一套与 backend
无关的接触计划。可先加 `--limit 10` 做快速验证，正式比较时再去掉该参数。

## 仓库结构

~~~text
anydexretarget/
  hand_representation.py        CanonicalGraspState
  optimizer/                    AnyDex 优化后端
assets/
  linkerhand_l25/               L25 URDF、MuJoCo 与 mesh
  linkerhand_l6/                L6 模型
  linkerhand_o6/                O6 模型
example/config/                 机器人、输入源和优化器配置
tools/
  capture_orbbec_rgbd.py        交互 RGB-D 采集
  generate_hug_candidates.py    HUG MANO 多候选生成
  benchmark_l25_retarget_backends.py
  rerank_l25_object_relative_candidates.py
  refine_l25_collision_aware.py
  build_l25_object_relative_scene.py
  l25_hardware_execute.py
docs/                           已验证 CLI 与集成文档
tests/                          中间态和主链路测试
~~~

## 当前限制

- Hunyuan 补全是学习得到的几何先验，Hybrid 点云不保证对所有物体都优于单帧真实深度。
- 接触目标在每根 MANO 远端指骨上采样三个位置作为指腹代理点，尚不是稠密的 MANO 压力接触 patch。
- 双指捏取使用“两接触点 + 手腕”完成刚体对齐；三指和包络抓取使用全部有效接触代理点。
- MuJoCo 优化同时惩罚物体穿透和不同手指之间的自碰撞，但效果仍依赖碰撞 mesh 的质量。
- 当前轨迹只是重复一个静态 qpos，尚未生成接近、闭合、力控和释放动作。
- 本项目不控制机械臂，也不负责将相机坐标系中的物体位姿转换到机器人基座坐标系。
- 当前没有 force closure 证明、触觉反馈闭环或通用真机抓取成功保证。

## 文档

- [完整已验证 CLI](docs/cli_reference.md)
- [HUG 接入与表示边界](docs/hug_retargeting.md)
- [Orbbec Gemini 335 采集](docs/orbbec_gemini335_capture.md)
- [CAD 或 mesh 输入 HUG](docs/cad_mesh_to_hug.md)
- [L25 Pico 4 仿真](docs/l25_pico4_simulation.md)

## 上游、分支与许可证

- **main**：Universal Retarget 项目代码与当前文档。
- **upstream-history**：完整 AnyDexRetarget 上游历史。
- **upstream**：[qqsq12321/AnyDexRetarget](https://github.com/qqsq12321/AnyDexRetarget)。

仓库保留上游 [MIT License](LICENSE)。HUG、Hunyuan3D、MANO、dex-retargeting、LinkerHand 模型与厂商 SDK 分别受各自许可证约束。

## 安全说明

真机执行可能损坏机械手、物体或周边设备。执行前必须在 MuJoCo 中检查完全相同的 qpos，先运行只读预检，清空现场，使用保守速度和力矩，并确保紧急停止手段可用。
