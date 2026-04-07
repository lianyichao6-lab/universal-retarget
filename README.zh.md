中文 | [English](README.md)

# AnyDexRetarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

高精度手部姿态重定向系统。支持两种优化器（**Adaptive** 和 **KeyVector**），多种灵巧手模型和多种手部追踪输入源，可用于仿真与遥操作。

## 演示

### 仿真重定向

https://github.com/user-attachments/assets/0950b2b0-ecd4-4270-abf6-5729dc05c6cb

### Quest 3 手臂遥操作

https://github.com/user-attachments/assets/4bcac46b-a603-4c0c-9d70-83d4351c9811

## 特性

- **Shadow Hand 支持**：Shadow Hand + MuJoCo Menagerie 高精度模型
- **两种优化器**：`adaptive`（对指感知，默认）和 `vector`（关键向量匹配）
- **高精度对指**：自适应优化，精确的拇指-手指接触
- **实时性能**：解析梯度 + NLopt SLSQP（~2ms/帧）
- **多输入源**：Apple Vision Pro、Meta Quest 3、笔记本摄像头（MediaPipe）、录制数据回放

## 目录

- [支持的机器人](#支持的机器人)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [优化器详解](#优化器详解)
- [引用](#引用)
- [致谢](#致谢)
- [联系方式](#联系方式)

## 支持的机器人

配置文件按**优化器类型**和**输入源**分类：

```
example/config/
├── adaptive/          # AdaptiveOptimizerAnalytical（默认）
│   ├── mediapipe/     # 摄像头 / 视频 / 回放
│   ├── avp/           # Apple Vision Pro
│   └── quest3/        # Meta Quest 3
└── vector/            # KeyVectorOptimizer
    ├── mediapipe/
    ├── avp/
    └── quest3/
```

| 机器人 | `--robot` 参数 | 配置后缀 | 说明 |
|--------|----------------|----------|------|
| **Shadow Hand** | `shadow` | `shadow_hand` | Shadow Hand + MuJoCo Menagerie 模型（默认仿真目标） |
| **Wuji Hand** | `wuji` | `wuji_hand` | 无极灵巧手，5 指 / 20 自由度 |
| **Allegro Hand** | `allegro` | `allegro_hand` | Allegro Hand，4 指 / 16 自由度 |
| **Inspire Hand** | `inspire` | `inspire_hand` | 因时灵巧手，含 mimic 关节 |
| **Ability Hand** | `ability` | `ability_hand` | Ability Hand，含 mimic 关节 |
| **Leap Hand** | `leap` | `leap_hand` | Leap Hand，4 指 / 16 自由度 |
| **SVH Hand** | `svh` | `svh_hand` | Schunk SVH Hand，含 mimic 关节 |
| **LinkerHand L21** | `linkerhand_l21` | `linkerhand_l21` | LinkerHand L21 |
| **ROHand** | `rohand` | `rohand` | ROHand |
| **Unitree Dex5** | `unitree_dex5` | `unitree_dex5_hand` | Unitree Dex5 |

> vector 优化器目前提供 `shadow_hand` 和 `wuji_hand` 的配置，其他机器人仅支持 `adaptive`。

## 仓库结构

```text
├── anydexretarget/
│   ├── retarget.py                        # 高层统一接口
│   ├── robot.py                           # Pinocchio 机器人包装
│   ├── mediapipe.py                       # MediaPipe 坐标变换
│   └── optimizer/                         # 优化器实现
│       ├── base_optimizer.py              # 基础优化器（FK/雅可比）
│       ├── analytical_optimizer.py        # AdaptiveOptimizerAnalytical
│       ├── key_vector_optimizer.py        # KeyVectorOptimizer
│       ├── robot_configs.py               # 机器人 link/URDF 配置
│       └── utils.py                       # TimingStats, LPFilter, Huber 损失
├── example/
│   ├── teleop_sim.py                      # MuJoCo 仿真示例
│   ├── teleop_real.py                     # 真机控制
│   ├── input/                             # 输入设备模块
│   │   ├── landmark_utils.py              # 共享 MediaPipe 关键点处理
│   │   ├── camera.py / video.py / ...     # 各输入设备
│   ├── test/                              # 调试与可视化工具
│   ├── config/
│   │   ├── adaptive/                      # AdaptiveOptimizerAnalytical 配置
│   │   │   ├── avp/                       # Apple Vision Pro
│   │   │   ├── quest3/                    # Meta Quest 3
│   │   │   └── mediapipe/                 # MediaPipe（摄像头/视频/回放）
│   │   └── vector/                        # KeyVectorOptimizer 配置
│   │       ├── avp/
│   │       ├── quest3/
│   │       └── mediapipe/
│   └── data/                              # 示例录制数据
├── assets/                                # 机器人 URDF / MuJoCo 资源
└── requirements.txt
```

## 安装

### 环境要求

- Python >= 3.10
- （可选）Apple Vision Pro + [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用
- （可选）Meta Quest 3 + [Hand Tracking Streamer](https://github.com/wengmister/hand-tracking-streamer) 应用

### 安装步骤

```bash
git clone https://gitee.com/gx_robot/AnyDexRetarget.git
cd AnyDexRetarget

# 通过 conda 安装 pinocchio（推荐，预编译二进制包）
conda install -c conda-forge pinocchio

# 安装其他依赖
pip install -r requirements.txt
pip install -e .
```

### 故障排除

**pinocchio 安装问题**：`pinocchio` 需要通过 conda 安装（不要用 pip）。pip 上的 `pin` 包需要 C++ 编译，在 Windows 上通常会失败：
```bash
conda install -c conda-forge pinocchio
```

**macOS MuJoCo**：仿真脚本使用 `mjpython` 代替 `python`：
```bash
mjpython example/teleop_sim.py --video example/data/right.mp4
```

## 快速开始

仓库当前自带两个示例输入：

- `example/data/right.mp4`：示例视频输入
- `example/data/avp1.pkl`：可选的录制回放输入

### 仿真

```bash
cd example

# 运行仓库自带示例视频（adaptive 优化器，默认）
python teleop_sim.py --video data/right.mp4 --robot shadow --hand right

# 切换到 KeyVector 优化器
python teleop_sim.py --video data/right.mp4 --robot shadow --hand right --optimizer vector

# 回放可选示例录制数据
python teleop_sim.py --play data/avp1.pkl --robot shadow --hand right

# 笔记本摄像头实时遥操作（MediaPipe）
python teleop_sim.py --input camera --robot shadow --hand right

# Vision Pro 实时遥操作
python teleop_sim.py --input visionpro --robot shadow --ip <vision-pro-ip> --hand right

# Quest 3 实时遥操作（通过 Hand Tracking Streamer）
python teleop_sim.py --input quest3 --robot shadow --port 9000 --hand right

# RealSense 实时遥操作
python teleop_sim.py --realsense --robot shadow --hand right --show-video

# 回放你自己的录制文件（.pkl）
python teleop_sim.py --play path/to/record.pkl --robot shadow --hand right
```

### 真机控制

`teleop_real.py` 以 **Wuji Hand** 为示例，演示真机遥操作。它通过 `wujihandpy` 发送 `5 x 4` 关节目标，你可以参考其控制循环适配其他灵巧手。

```bash
cd example

# Vision Pro -> Wuji Hand（adaptive）
python teleop_real.py --robot wuji --input visionpro --ip <vision-pro-ip> --hand right

# Vision Pro -> Wuji Hand（vector 优化器）
python teleop_real.py --robot wuji --input visionpro --ip <vision-pro-ip> --hand right --optimizer vector

# 回放可选示例录制数据 -> Wuji Hand
python teleop_real.py --robot wuji --play data/avp1.pkl --hand right

# Linux USB 权限
sudo chmod a+rw /dev/ttyUSB0
```

### 命令参考

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--robot` | `shadow`（sim）/ `wuji`（real） | 灵巧手类型 |
| `--optimizer` | `adaptive` | 优化器类型：`adaptive` 或 `vector` |
| `--config` | 自动选择 | 配置文件（覆盖 `--robot` 和 `--optimizer`） |
| `--hand` | `right` | 手的方向（`left`/`right`） |
| `--input` | - | `teleop_sim.py`：`visionpro` / `quest3` / `camera` / `realsense` / `video` / `mediapipe_replay` |
| `--input` | - | `teleop_real.py`：`visionpro` / `mediapipe_replay` |
| `--realsense` | 关闭 | `--input realsense` 的快捷方式 |
| `--play FILE` | - | 回放录制（`--input mediapipe_replay` 的快捷方式） |
| `--video FILE` | - | 视频文件输入（MediaPipe 手部检测） |
| `--ip` | `192.168.50.127` | Vision Pro IP |
| `--port` | `9000` | Quest 3 HTS 监听端口 |
| `--protocol` | `udp` | Quest 3 HTS 传输协议（`udp`/`tcp`） |
| `--speed` | `1.0` | 播放速度 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出文件路径 |
| `--show-video` | 关闭 | 显示 RGB / 关键点预览 |
| `--video-depth-scale` | `1.25` | `--video` 模式下额外深度缩放 |
| `--no-loop` | - | 禁用回放循环 |
| `--headless` | 关闭 | 无 GUI 运行仿真 |
| `--save-sim FILE` | - | 保存离屏仿真视频 |
| `--save-qpos FILE` | - | 保存目标 / 仿真 qpos 轨迹 |

### 调试与可视化工具

#### debug_skeleton.py

在 MuJoCo 查看器中对比三套骨架，用于调试重定向问题：

- **蓝色**：原始 MediaPipe 骨架（坐标变换后，未缩放）
- **绿色**：缩放后的目标骨架（优化器的匹配目标）
- **红色**：机器人 FK 骨架（重定向结果）

```bash
cd example

# 摄像头输入
python test/debug_skeleton.py --robot leap --input camera

# 视频文件输入
python test/debug_skeleton.py --robot leap --video data/right.mp4

# 使用可选示例录制数据，对比两种优化器
python test/debug_skeleton.py --robot shadow --play data/avp1.pkl --optimizer adaptive
python test/debug_skeleton.py --robot shadow --play data/avp1.pkl --optimizer vector

# 你自己的录制数据
python test/debug_skeleton.py --robot shadow --play path/to/record.pkl
```

#### visualize_scaling.py

可视化 `scaling` 和 `segment_scaling` 参数对 MediaPipe 关键点的影响。

```bash
cd example

python test/visualize_scaling.py --robot leap --video data/right.mp4 --hand right
python test/visualize_scaling.py --robot allegro --play data/avp1.pkl --hand right
```

## 配置说明

### Adaptive 优化器配置

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"

robot:
  type: "shadow_hand"

retarget:
  # 损失权重
  w_pos: 1.0              # 指尖位置权重
  w_dir: 5.0              # 指尖方向权重
  w_full_hand: 1.0        # 全手权重

  # Huber 损失阈值
  huber_delta: 2.0        # 位置阈值（cm）
  huber_delta_dir: 0.5    # 方向阈值

  # 正则化
  norm_delta: 0.04        # 速度平滑

  # 缩放
  scaling: 0.81           # MediaPipe 到机器人缩放（Shadow Hand 约为 MediaPipe 的 81%）

  # 坐标系对齐
  mediapipe_rotation:
    x: 0.0
    y: 0.0
    z: -90.0              # Shadow Hand 需要 -90° Z 旋转

  # 对指阈值（cm）
  pinch_thresholds:
    index:  { d1: 2.0, d2: 4.0 }
    middle: { d1: 2.0, d2: 4.0 }
    ring:   { d1: 2.0, d2: 4.0 }
    pinky:  { d1: 2.0, d2: 4.0 }

  # 低通滤波器（0~1，越小越平滑）
  lp_alpha: 0.4
```

### KeyVector 优化器配置

```yaml
optimizer:
  type: "KeyVectorOptimizer"

robot:
  type: "wuji_hand"
  urdf_path: "assets/wuji_hand/right.urdf"

retarget:
  huber_delta: 2.0
  norm_delta: 0.04
  lp_alpha: 0.4

  # 每项：机器人（origin_link -> task_link）匹配 MediaPipe（origin_kp -> task_kp）关键点对
  # scale：将人手向量缩放以匹配机器人手部尺寸
  key_vectors:
    - {origin: right_palm_link, task: right_finger1_link3,    origin_kp: 0, task_kp:  2, scale: 1.0}
    - {origin: right_palm_link, task: right_finger1_link4,    origin_kp: 0, task_kp:  3, scale: 1.0}
    - {origin: right_palm_link, task: right_finger1_tip_link, origin_kp: 0, task_kp:  4, scale: 1.0}
    # ... （共 15 个向量：每指 3 个）

  mediapipe_rotation:
    x: -5.0
    y: -5.0
    z: 0.0
```

### 关键参数

| 参数 | 优化器 | 说明 |
|------|--------|------|
| `scaling` | adaptive | 手部尺寸比例。Shadow Hand ≈ 0.81 |
| `w_pos` / `w_dir` / `w_full_hand` | adaptive | 损失项权重 |
| `pinch_thresholds` | adaptive | 对指检测距离阈值（cm） |
| `key_vectors[i].scale` | vector | 每向量缩放以匹配机器人手部尺寸 |
| `mediapipe_rotation` | 两者 | 坐标系对齐（角度） |
| `norm_delta` | 两者 | 速度正则化权重 |
| `lp_alpha` | 两者 | 低通滤波系数（0~1） |

## API 参考

### 基本用法

```python
from anydexretarget import Retargeter

# 从配置文件加载
retargeter = Retargeter.from_yaml("config/adaptive/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")

# 重定向：(21, 3) MediaPipe 关键点 -> 关节角度
qpos = retargeter.retarget(raw_keypoints)

# 带详细输出
qpos, info = retargeter.retarget_verbose(raw_keypoints)
print(f"Cost: {info['cost']:.4f}")
print(f"Pinch alphas: {info.get('pinch_alphas')}")  # 仅 adaptive
```

### 高级用法

```python
# 直接访问优化器
optimizer = retargeter.optimizer

# 计算给定姿态的代价
cost = optimizer.compute_cost(qpos, mediapipe_keypoints)

# 获取计时统计
stats = optimizer.get_timing_stats()
print(f"平均耗时: {stats.get_avg()['total_ms']:.2f} ms")
```

## 优化器详解

### AdaptiveOptimizerAnalytical

基于拇指到手指距离，在 TipDirVec（对指）和 FullHandVec（张开）之间自适应混合。

**损失函数：**
```
L = Σᵢ [αᵢ · (w_pos·L_pos + w_dir·L_dir) + (1-αᵢ) · w_full·L_full] + norm_delta·||Δq||²
```

**自适应混合：**
```
αᵢ = 0.7    如果 dᵢ < d1  (对指 → TipDirVec 模式)
αᵢ = 0.0    如果 dᵢ > d2  (张开 → FullHandVec 模式)
αᵢ = 插值   其他情况
```

其中 `dᵢ` 是拇指到手指指尖的距离。

### KeyVectorOptimizer

最小化 N 个机器人 link 对向量与对应的缩放后 MediaPipe 关键点向量之间的平均 Huber 距离。参考：[dex-retargeting](https://github.com/dexsuite/dex-retargeting) VectorOptimizer。

**损失函数：**
```
L = (1/N) · Σᵢ Huber(‖[FK(task_i) - FK(origin_i)] - scale_i·(mp[task_kp_i] - mp[origin_kp_i])‖)
  + norm_delta · ‖q - q_prev‖²
```

**适用场景：**
- 行为更简单、更可预测
- 无对指检测——适合不需要精确手指接触的任务
- 易于自定义：在 YAML 中增删向量或调整每向量缩放

## 引用

```bibtex
@software{anydexretarget2025,
  title={AnyDexRetarget},
  author={Shiquan Qiu},
  year={2025},
  url={https://gitee.com/gx_robot/AnyDexRetarget},
}
```

## 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Shadow Hand 模型
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) - 重定向算法
- [DexPilot](https://arxiv.org/abs/1910.03135) - 基于视觉的遥操作
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) - Apple Vision Pro 数据流
- [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) - 无极重定向

## 联系方式

如有问题，请在 [Gitee](https://gitee.com/gx_robot/AnyDexRetarget/issues) 上提交 issue 或通过 932851972@qq.com 联系作者。
