# vla_nav
Long-Horizon Social Navigation for Human-Centric Outdoor Assistance
# CityWalker 自主导航系统 — ROS 路径规划包

> 基于视觉-语言-动作模型（VLA）与 LLM 的自主导航系统，运行于 **Unitree Go2 四足机器人**，部署于 **中国科学技术大学高新校区**。

---

## 📋 项目概述

本项目是一个完整的机器人自主导航系统，结合了三种导航方式：

| 方式 | 核心模块 | 说明 |
|------|----------|------|
| 🧠 **LLM 路径规划** | `llm_gps_planner.py` | 通过大语言模型理解自然语言指令，结合 POI 数据库规划全局路径 |
| 👁️ **VLA 视觉导航** | `vla_navigation_node.py` + `loadmodel.py` + `data_collector.py` | 使用 CityWalker 模型，基于历史图像帧与里程计进行局部轨迹预测 |
| 🛰️ **GPS 全局导航** | `odom_gps_aligner.py` + `alignment_motion.py` | 通过 GPS（WGS84）与轮式里程计对齐，实现全局定位与路径跟踪 |

系统架构：

```
用户指令 / 全局路径
       │
       ▼
┌─────────────────────┐     ┌──────────────────────┐
│  LLM 路径规划       │ ──▶ │  AMap 路径规划 API    │
│  (llm_gps_planner)  │     │  (amap_route.html)    │
└─────────┬───────────┘     └──────────────────────┘
          │ 全局途经点 (GCJ-02)
          ▼
┌─────────────────────────────────────────────┐
│  VLA 导航控制节点                            │
│  (vla_navigation_node)                      │
│  ┌──────────────────────────────────────┐   │
│  │  TimeWindowCache (data_collector)    │   │
│  │  ← 实时缓存 5s 图像 + 里程计          │   │
│  ├──────────────────────────────────────┤   │
│  │  CityWalker 模型推理 (loadmodel)     │   │
│  │  ← 预测未来 5 个轨迹点               │   │
│  ├──────────────────────────────────────┤   │
│  │  坐标变换 (coord_transform_robot)    │   │
│  │  ← WGS84 / GCJ-02 / Odom 坐标转换    │   │
│  ├──────────────────────────────────────┤   │
│  │  TargetPointController (target_point)│   │
│  │  ← 闭环控制机器人到达目标点          │   │
│  └──────────────────────────────────────┘   │
└─────────────────────┬───────────────────────┘
                      │ 速度/转向指令
                      ▼
           ┌──────────────────┐
           │  Unitree Go2     │
           │  四足机器人       │
           └──────────────────┘
```

---

## 🗂️ 文件说明

### 核心节点

| 文件 | 功能 |
|------|------|
| `vla_navigation_node.py` | **主导航控制节点** — 统筹全局路径遍历、数据采集、模型推理、坐标变换与底层控制 |
| `llm_gps_planner.py` | **LLM 路径规划节点** — 通过大语言模型理解用户指令，结合 POI 与高德地图 API 规划全局路径 |
| `odom_gps_aligner.py` | **Odom-GPS 对齐节点** — 计算轮式里程计与 GPS 之间的航向偏差角 |
| `alignment_motion.py` | **对齐模式运动控制** — 对齐阶段控制机器人匀速直走并保持航向 |
| `target_point.py` | **局部轨迹点执行控制器** — 闭环控制机器人到达目标点（带 yaw 朝向控制） |
| `data_collector.py` | **时间窗口数据缓存器** — 实时缓存近 10s 的里程计与图像数据，按时间回溯采样 |
| `loadmodel.py` | **CityWalker 模型推理引擎** — 预加载 CityWalkerFeat 模型到 GPU，提供统一推理接口 |
| `hello.py` | **数据可视化示例** — 实时可视化 TimeWindowCache 输出的 5 帧图像与相对位姿 |

### 工具模块

| 文件 | 功能 |
|------|------|
| `coord_transform_robot.py` | **坐标转换工具** — WGS84↔GCJ-02↔UTM 坐标转换，机器人坐标系下目标位姿计算 |
| `coordTransform_utils.py` | **开源坐标转换库** — WGS84 / GCJ-02 / BD-09 互转（GCJ-02 加密坐标系工具） |
| `pub_odom.py` | **仿真数据发布器** — 在没有机器人硬件时模拟发布 `/odom` 和 `/fix` 话题 |
| `visualpath.py` | **路径可视化数据** — 预定义的全局路径途经点坐标（GCJ-02 格式） |

### 配置与数据

| 文件 | 功能 |
|------|------|
| `ustc_gaoxin_poi.json` | **POI 数据库** — 中科大高新校区关键兴趣点（建筑、大门、运动场馆等） |
| `ustc_gaoxin_poi.py` | POI 数据查看/转换脚本 |
| `amap_route.html` | **高德地图路径规划** — 在浏览器中展示 LLM 规划的路线 |

---

## 🔧 环境依赖

### 硬件

- **机器人**: Unitree Go2（四足机器人）
- **计算单元**: 机载计算机（建议配置 GPU 以运行 CityWalker 模型）
- **传感器**: 摄像头（前后/全景）、GPS 模块

### 软件

| 依赖 | 说明 |
|------|------|
| ROS (Noetic) | 机器人操作系统 |
| Python 3.8+ | 运行环境 |
| PyTorch | CityWalker 模型推理框架 |
| OpenCV | 图像处理 |
| Unitree SDK2 | Go2 机器人底层控制接口 |
| CityWalker 模型 | [CityWalker](https://github.com/KuanchihHuang/CityWalker) 视觉导航模型 |
| DashScope API | 通义千问 LLM 接口（用于 LLM 路径规划） |
| 高德地图 API | 路径规划与地理编码服务 |

### Python 包

```
rospy
torch
numpy
opencv-python
scipy
pyproj
requests
openai                 # DashScope 兼容 OpenAI SDK
cv_bridge
unitree_sdk2py         # Unitree 机器人 SDK
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确认 ROS 环境已配置
source /opt/ros/noetic/setup.bash

# 安装 Python 依赖
pip install torch numpy opencv-python scipy pyproj requests openai

# 安装 Unitree SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

### 2. 配置 CityWalker 模型

将 CityWalker 检查点放置于：

```
/home/robot/CityWalker/checkpoints/
├── CityWalker_2000hr.ckpt
└── dinov2_vitb14_pretrain.pth
```

在 `loadmodel.py` 中调整参数：

```python
STEP_SCALE = 0.60        # 步长缩放因子（根据机器人实际速度调整）
IMG_SIZE = (350, 630)    # 输入图像尺寸 (H, W)
```

### 3. 配置 API 密钥

在 `llm_gps_planner.py` 中填入你的密钥：

```python
AMAP_KEY = "你的高德地图API Key"
client = OpenAI(
    api_key="你的DashScope API Key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

### 4. 运行系统

```bash
# 终端1：启动 ROS Master
roscore

# 终端2：启动 Odom-GPS 对齐（确定航向偏差）
rosrun pathplanning odom_gps_aligner.py enp58s0

# 终端3：启动对齐运动控制
rosrun pathplanning alignment_motion.py enp58s0

# 终端4：启动 LLM 路径规划（或直接发布全局路径）
rosrun pathplanning llm_gps_planner.py

# 终端5：启动 VLA 导航主节点
rosrun pathplanning vla_navigation_node.py

# 终端6（可选）：实时可视化缓存数据
rosrun pathplanning hello.py

# 仿真模式（无机器人硬件）
rosrun pathplanning pub_odom.py
```

### 5. 发布导航指令

```bash
# 通过 LLM（自然语言）
rostopic pub /user_instruction std_msgs/String "data: '带我去图书馆'"

# 或发布全局路径
rostopic pub /planned_path nav_msgs/Path ...
```

---

## 🧩 核心模块详解

### 🧠 LLM 路径规划 (`llm_gps_planner.py`)

- 接收自然语言指令 → LLM 解析目标地点
- 匹配 POI 数据库 → 获取目标 GPS 坐标
- 调用高德地图 API → 获取步行/驾车路径
- 发布全局导航路径 (`/planned_path`)

### 👁️ CityWalker VLA 导航 (`vla_navigation_node.py`)

导航循环：
1. **数据采集**: `TimeWindowCache` 从 ROS 话题缓存近 10s 的里程计与图像
2. **时间回溯采样**: 以 `T-4, T-3, T-2, T-1, T0` 间隔采样 5 帧图像 + 相对位姿
3. **坐标变换**: 将全局目标点转换到机器人坐标系
4. **模型推理**: CityWalker 根据历史 5 帧预测未来 5 个轨迹点（机器人坐标系）
5. **轨迹执行**: 将轨迹点转换到世界坐标系 → 逐点发送给 `target_point` 执行
6. **到达判定**: 到达全局途经点 → 切换下一目标；未到达 → 重新采集推理

### 🛰️ Odom-GPS 对齐 (`odom_gps_aligner.py` + `alignment_motion.py`)

1. 机器人匀速直走 5m
2. 记录起始与结束的 GPS(UTM) 与 odom 位姿
3. 计算航向偏差角 `yaw_correction = gps_angle - odom_angle`
4. 持续发布 `/yaw_alignment` 供导航层校准

### 🎯 轨迹点执行 (`target_point.py`)

- 接收世界坐标系下的目标点
- P 控制器生成 yaw 速度，使机器人朝向目标
- 到达（距离 < 阈值）或超时（30s）判定
- 通过 Unitree SDK 控制 Go2 机器人运动

---

## 📐 坐标系约定

| 坐标系 | 描述 |
|--------|------|
| `WGS84` | GPS 标准经纬度坐标 (`/fix` 话题) |
| `GCJ-02` | 高德地图使用的加密坐标系 |
| `UTM Zone 50N` | 合肥地区 UTM 投影坐标（米） |
| `odom` | ROS 里程计坐标系（全局） |
| `robot` | 机器人坐标系：x 前、y 左、yaw=0 为车头朝正东 |

---

## 🧪 测试与调试

### 仿真模式

在没有机器人硬件时，使用 `pub_odom.py` 模拟发布 `/odom` 和 `/fix` 话题：

```bash
rosrun pathplanning pub_odom.py
```

### 可视化

`hello.py` 启动 OpenCV 窗口，实时显示 TimeWindowCache 的 5 帧采样图像及相对位姿信息：

```bash
rosrun pathplanning hello.py
```

### 路网点查看

```bash
rosrun pathplanning ustc_gaoxin_poi.py
```

---

## 📄 许可证

本项目基于 BSD 3-Clause 许可证，详见上层目录的 `LICENSE` 文件。

> 注意：本项目包含的子模块（CityWalker 模型、Unitree SDK、高德地图 API）遵循各自的许可协议。

---

## 🙏 致谢

- [CityWalker](https://github.com/KuanchihHuang/CityWalker) — 视觉语言导航模型
- 中国科学技术大学 — 测试场地与技术支持
- 高德开放平台 — 地图 API 服务
- 阿里云 DashScope — LLM API 服务
