中文 | [English](README.md)

# qsq-retargeting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

高精度手部姿态重定向系统。基于自适应解析优化，支持 Shadow Hand（MuJoCo Menagerie），支持 Apple Vision Pro 手部追踪实时遥操作。

## 演示

[![演示视频](assets/demo_cover.png)](https://www.bilibili.com/video/BV1YfFXzXEwr)

## 特性

- **Shadow Hand 支持**：Shadow Hand + MuJoCo Menagerie 高精度模型
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

| 机器人 | 配置文件 | 说明 |
|--------|----------|------|
| **Shadow Hand** | `shadow_hand_menagerie.yaml` | Shadow Hand + MuJoCo Menagerie 高精度模型（默认） |
| **Wuji Hand** | `wuji_hand.yaml` | 无极灵巧手 5 指 20 自由度 |
| **Allegro Hand** | `allegro_hand.yaml` | Allegro Hand 4 指 16 自由度 |
| **Inspire Hand** | `inspire_hand.yaml` | 因时灵巧手 5 指（含 mimic 关节） |
| **Ability Hand** | `ability_hand.yaml` | Ability Hand 5 指（含 mimic 关节） |
| **Leap Hand** | `leap_hand.yaml` | Leap Hand 4 指 16 自由度 |
| **SVH Hand** | `svh_hand.yaml` | Schunk SVH Hand 5 指（含 mimic 关节） |

## 仓库结构

```text
├── qsq_retargeting/
│   ├── retarget.py                        # 高层统一接口
│   ├── robot.py                           # Pinocchio 机器人包装
│   ├── mediapipe.py                       # MediaPipe 坐标变换
│   └── optimizer/                         # 优化器实现
│       ├── base_optimizer.py              # 基础优化器（FK/雅可比）
│       ├── analytical_optimizer.py        # 自适应优化器（解析梯度）
│       ├── robot_configs.py               # 机器人 link/URDF 配置
│       └── utils.py                       # TimingStats, LPFilter, Huber 损失
├── example/
│   ├── teleop_sim.py                      # MuJoCo 仿真示例
│   ├── teleop_real.py                     # 真机控制
│   ├── input/                             # 输入设备模块
│   │   ├── landmark_utils.py              # 共享 MediaPipe 关键点处理
│   │   ├── camera.py / video.py / ...     # 各输入设备
│   ├── test/                              # 调试与可视化工具
│   ├── config/                            # YAML 配置文件（按输入源分类）
│   │   ├── avp/                           # Apple Vision Pro 配置
│   │   ├── quest3/                        # Meta Quest 3 配置
│   │   └── mediapipe/                     # MediaPipe（摄像头/视频/回放）配置
│   └── data/                              # 示例录制数据
└── requirements.txt
```

## 安装

### 环境要求

- Python >= 3.10
- （可选）Apple Vision Pro + [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用

### 安装步骤

```bash
git clone https://gitee.com/gx_robot/qsq-retargeting.git
cd qsq-retargeting

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
mjpython example/teleop_sim.py --play example/data/avp1.pkl
```

## 快速开始

### Shadow Hand（默认）

```bash
cd example

# 回放录制数据
python teleop_sim.py --play data/avp1.pkl --hand right

# 笔记本摄像头实时遥操作（MediaPipe）
python teleop_sim.py --input camera --hand right

# Vision Pro 实时遥操作
python teleop_sim.py --input visionpro --ip <vision-pro-ip> --hand right

# Quest 3 实时遥操作（通过 Hand Tracking Streamer）
python teleop_sim.py --input quest3 --port 9000 --hand right
```

### 真机控制

```bash
cd example
python teleop_real.py --play data/avp1.pkl --hand right

# Linux USB 权限
sudo chmod a+rw /dev/ttyUSB0
```

### 命令参考

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--robot` | `shadow` | 灵巧手类型（如 `wuji`、`allegro`、`inspire`） |
| `--config` | 自动选择 | 配置文件（覆盖 `--robot`） |
| `--hand` | `left`（sim）/ `right`（real） | 手的方向（`left`/`right`） |
| `--input` | - | 输入类型（`visionpro`/`quest3`/`camera`/`mediapipe_replay`） |
| `--play FILE` | - | 回放录制（`--input mediapipe_replay` 的快捷方式） |
| `--video FILE` | - | 视频文件输入（MediaPipe 手部检测） |
| `--ip` | `192.168.50.127` | Vision Pro IP |
| `--port` | `9000` | Quest 3 HTS 监听端口 |
| `--protocol` | `udp` | Quest 3 HTS 传输协议（`udp`/`tcp`） |
| `--speed` | `1.0` | 播放速度 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出文件路径 |
| `--no-loop` | - | 禁用回放循环 |

### 调试与可视化工具

#### debug_skeleton.py

在 MuJoCo 查看器中对比三���骨架，用于调试重定向问题：

- **蓝色**：原始 MediaPipe 骨架（坐标变换后，未缩放）
- **绿色**：缩放后的目标骨架（优化器的匹配目标）
- **红色**：机器人 FK 骨架（重定向结果）

```bash
cd example

# 摄像头输入
python test/debug_skeleton.py --robot leap --input camera

# 视频文件输入
python test/debug_skeleton.py --robot leap --video data/right.mp4

# 回放录制数据
python test/debug_skeleton.py --robot shadow --play data/avp1.pkl
```

#### visualize_scaling.py

可视化 `scaling` 和 `segment_scaling` 参数对 MediaPipe 关键点的影响。在 matplotlib 3D 图中对比原始骨架和缩放后的目标骨架。

```bash
cd example

# 回放录制数据
python test/visualize_scaling.py --robot allegro --play data/avp1.pkl --hand right

# 视频文件输入
python test/visualize_scaling.py --robot leap --video data/right.mp4 --hand right
```

## 配置说明

### 配置文件结构

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"

robot:
  type: "shadow_hand_menagerie"

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

### 关键参数

| 参数 | 说明 |
|------|------|
| `scaling` | 手部尺寸比例。Shadow Hand ≈ 0.81 |
| `mediapipe_rotation.z` | 坐标系对齐。Shadow Hand = -90° |

## API 参考

### 基本用法

```python
from qsq_retargeting import Retargeter

# 从配置文件加载
retargeter = Retargeter.from_yaml("config/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")

# 重定向：(21, 3) MediaPipe 关键点 -> 关节角度
qpos = retargeter.retarget(raw_keypoints)

# 带详细输出
qpos, info = retargeter.retarget_verbose(raw_keypoints)
print(f"Cost: {info['cost']:.4f}")
print(f"Pinch alphas: {info['pinch_alphas']}")
```

### 高级用法

```python
# 直接访问优化器
optimizer = retargeter.optimizer

# 计算给定姿态的代价
cost = optimizer.compute_cost(qpos, mediapipe_keypoints)

# 获取计时统计
stats = optimizer.get_timing_stats()
print(f"平均耗时: {stats.avg_total_ms:.2f} ms")
```

## 优化器详解

### 优化公式

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

### 损失函数

```
L = Σᵢ [αᵢ · L_tip_dir_vec + (1-αᵢ) · L_full_hand] + norm_delta · ||Δq||²
```

- **L_tip_dir_vec**：位置 + 方向匹配（用于对指手势）
- **L_full_hand**：全手向量匹配（用于张开手势）

### 自适应混合

```
αᵢ = 0.7    如果 dᵢ < d1  (对指 → TipDirVec 模式)
αᵢ = 0.0    如果 dᵢ > d2  (张开 → FullHandVec 模式)
αᵢ = 插值   其他情况
```

其中 `dᵢ` 是拇指到手指的距离。

## 引用

```bibtex
@software{qsq2025retargeting,
  title={Hand Retargeting},
  author={QSQ},
  year={2025},
  url={https://gitee.com/gx_robot/qsq-retargeting},
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

如有问题，请在 [Gitee](https://gitee.com/gx_robot/qsq-retargeting/issues) 上提交 issue 或通过 932851972@qq.com 联系作者。
