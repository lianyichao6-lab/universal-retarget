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
- **多输入源**：Apple Vision Pro、笔记本摄像头（MediaPipe）、录制数据回放
- **肌腱耦合**：Shadow Hand J2-J1 耦合约束，兼容 MuJoCo 仿真

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
| **Shadow Hand Menagerie** | `shadow_hand_menagerie.yaml` | Shadow Hand + MuJoCo Menagerie 高精度模型（默认） |
| **Shadow Hand** | `shadow_hand.yaml` | 24自由度 Shadow 灵巧手 |

## 仓库结构

```text
├── qsq_retargeting/
│   ├── opt/                          # 优化器实现
│   │   ├── base.py                   # 基础优化器（FK/雅可比）
│   │   └── adaptive_analytical.py    # 自适应优化器（解析梯度）
│   └── shadow_hand_menagerie/        # Shadow Hand URDF（MuJoCo Menagerie）
│       ├── left_hand_mj.urdf
│       └── right_hand_mj.urdf
├── example/
│   ├── teleop_sim.py                 # MuJoCo 仿真示例
│   ├── teleop_real.py                # 真机控制
│   ├── input_devices/                # 输入设备模块
│   ├── config/                       # YAML 配置文件
│   └── data/                         # 示例录制数据
└── requirements.txt
```

## 安装

### 环境要求

- Python >= 3.10
- （可选）Apple Vision Pro + [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用

### 安装步骤

```bash
git clone https://github.com/qqsq12321/qsq-retargeting.git
cd qsq-retargeting
pip install -r requirements.txt
pip install -e .
```

### 故障排除

**pinocchio 安装问题**：如果 PyPI 镜像源安装失败：
```bash
pip install pin==3.8.0 -i https://pypi.org/simple
```

**macOS MuJoCo**：仿真脚本使用 `mjpython` 代替 `python`：
```bash
mjpython example/teleop_sim.py --play example/data/avp1.pkl
```

## 快速开始

### Shadow Hand Menagerie（默认）

```bash
cd example

# 回放录制数据
python teleop_sim.py --play data/avp1.pkl --hand right

# 笔记本摄像头实时遥操作（MediaPipe）
python teleop_sim.py --input camera --hand right

# Vision Pro 实时遥操作
python teleop_sim.py --input visionpro --ip <vision-pro-ip> --hand right
```

### 其他 Shadow Hand 配置

```bash
cd example

# 原版 Shadow Hand（无 Menagerie 模型）
python teleop_sim.py --config config/shadow_hand.yaml --play data/avp1.pkl --hand right
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
| `--config` | `config/adaptive_analytical_avp.yaml` | 配置文件 |
| `--hand` | `left`（sim）/ `right`（real） | 手的方向（`left`/`right`） |
| `--input` | - | 输入类型（`visionpro`/`camera`/`mediapipe_replay`） |
| `--play FILE` | - | 回放录制（`--input mediapipe_replay` 的快捷方式） |
| `--ip` | `192.168.50.127` | Vision Pro IP |
| `--speed` | `1.0` | 播放速度 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出文件路径 |
| `--no-loop` | - | 禁用回放循环 |

## 配置说明

### 配置文件结构

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"

robot:
  type: "shadow_hand_menagerie"  # 或 "shadow_hand"

retarget:
  # 损失权重
  w_pos: 1.0              # 指尖位置权重
  w_dir: 5.0              # 指尖方向权重
  w_pinch: 50.0           # 对指距离权重（越高对指越精确）
  w_coupling: 100.0       # 肌腱耦合权重（仅 Shadow Hand）
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
| `w_pinch` | 对指精度权重。精细抓取任务建议 50-100 |
| `w_coupling` | Shadow Hand 肌腱耦合。设置 100+ 以匹配 MuJoCo 仿真 |
| `scaling` | 手部尺寸比例。Shadow Hand ≈ 0.81 |
| `mediapipe_rotation.z` | 坐标系对齐。Shadow Hand = -90° |

## API 参考

### 基本用法

```python
from qsq_retargeting import Retargeter

# 从配置文件加载
retargeter = Retargeter.from_yaml("config/shadow_hand_menagerie.yaml", hand_side="right")

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
L = Σᵢ [αᵢ · L_tip_dir_vec + (1-αᵢ) · L_full_hand] + w_pinch · L_pinch + w_coupling · L_coupling
```

- **L_tip_dir_vec**：位置 + 方向匹配（用于对指手势）
- **L_full_hand**：全手向量匹配（用于张开手势）
- **L_pinch**：拇指-手指直接距离匹配
- **L_coupling**：Shadow Hand 肌腱 |J2 - J1| 惩罚

### 自适应混合

```
αᵢ = 1.0    如果 dᵢ < d1  (对指 → TipDirVec 模式)
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
  url={https://github.com/qqsq12321/qsq-retargeting},
}
```

## 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Shadow Hand 模型
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) - 重定向算法
- [DexPilot](https://arxiv.org/abs/1910.03135) - 基于视觉的遥操作
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) - Apple Vision Pro 数据流

## 联系方式

如有问题，请在 [GitHub](https://github.com/qqsq12321/qsq-retargeting/issues) 上提交 issue。
