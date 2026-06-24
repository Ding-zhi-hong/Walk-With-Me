[README.md](https://github.com/user-attachments/files/29282142/README.md)
# qwen_gps_nav
Long-Horizon Social Navigation for Human-Centric Outdoor Assistance
# testGPS — LLM + 高德地图 + GPS 导航系统

基于 ROS 的智能 GPS 导航系统，融合 **通义千问（Qwen）大模型** 与 **高德地图（AMap）API**，实现自然语言→目的地→坐标转换→GPS 导航的完整机器人自主导航链路。

## 📋 项目概览

```
自然语言指令（如"去西门"）
    │
    ▼
┌─────────────────────────────┐
│   llm_gps_planner.py        │ ← 调用 Qwen LLM 解析目的地, 高德API获取POI/路线
│                              │
│   target_point.py            │ ← 目标点数据模型
└──────────┬──────────────────┘
           ▼ (经纬度坐标)
┌─────────────────────────────┐
│   coord_transform_robot.py   │ ← 多坐标系之间的转换
│   coordTransform_utils.py    │    WGS-84 ↔ GCJ-02 ↔ BD-09
└──────────┬──────────────────┘
           ▼ (机器人坐标系下坐标)
┌─────────────────────────────┐
│   gps_waypoint_navigator.py  │ ← GPS 航点导航，发布 cmd_vel
│                              │
│   odom_gps_aligner.py        │ ← 里程计与GPS坐标对齐/校准
│   alignment_motion.py        │ ← 对齐运动控制
└──────────┬──────────────────┘
           ▼
     机器人底盘运动
```

## 🧩 模块说明

### 1. LLM + 高德地图规划器 (`llm_gps_planner.py`)

- 接收用户的自然语言导航指令
- 调用 **通义千问（Qwen）** 大模型解析目的地语义
- 通过 **高德地图 API**（地理编码 / POI 搜索 / 路径规划）获取目标 GPS 坐标与路径
- 生成可执行的航点序列

### 2. 坐标转换工具 (`coordTransform_utils.py` / `coord_transform_robot.py`)

提供中国常用的 GPS 坐标系之间的精确双向转换：

| 坐标系 | 说明 | 使用方 |
|--------|------|--------|
| **WGS-84** | 国际标准，GPS 原始输出 | 全球定位系统 |
| **GCJ-02** | 国测局坐标，高德/腾讯地图使用 | 高德 API 返回结果 |
| **BD-09** | 百度坐标，百度地图使用 | 百度系服务 |

`coord_transform_robot.py` 额外封装了机器人导航场景下的坐标系转换（经纬度 → 机器人局部坐标系）。

### 3. GPS 航点导航器 (`gps_waypoint_navigator.py`)

- 订阅 GPS 传感器数据（`/fix`）
- 按顺序执行航点队列
- 计算航向与距离，发布 `cmd_vel` 控制机器人运动
- 支持到达判断（距离阈值）

### 4. 里程计-GPS 对齐 (`odom_gps_aligner.py` / `alignment_motion.py`)

- 将 GPS 坐标与机器人里程计坐标系对齐
- 执行校准运动，计算转换参数
- 解决初始位置标定问题

### 5. 目标点管理 (`target_point.py`)

- 目标点的数据模型（经纬度、名称、描述等）
- 预设校园/园区 POI 数据加载

### 6. POI 数据 (`ustc_gaoxin_poi.json`)

中国科学技术大学高新校区（USTC Gaoxin Campus）的预定义兴趣点数据，可直接用于校内导航。

## 🚀 安装与配置

### 环境要求

- **ROS** — Melodic / Noetic / ROS2（根据你的 `.ros` 配置）
- **Python** ≥ 3.8
- **通义千问 API** — 需阿里云 DashScope 密钥
- **高德地图 API** — 需高德开放平台 Key

### 克隆并编译

```bash
# 将项目放入 ROS 工作空间
cd ~/catkin_ws/src
git clone https://github.com/<your-username>/testGPS.git

# 编译
cd ~/catkin_ws
catkin_make
# 或 catkin build
```

### 依赖安装

```bash
# Python 依赖
pip install rospy  # ROS 自带
pip install requests
pip install dashscope        # 通义千问 SDK
pip install pyproj           # 专业坐标转换（可选增强）
```

### 配置 API 密钥

创建配置文件或在环境变量中设置：

```bash
# ~/.bashrc 或 launch 文件中
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"   # 阿里云 DashScope 密钥
export AMAP_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxx"   # 高德地图 Web API Key
```

或在 `llm_gps_planner.py` 中直接配置（不推荐用于生产）：

```python
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxx"
AMAP_API_KEY      = "xxxxxxxxxxxxxxxxxxxxxxxxx"
```

> **高德 Key 申请**：https://lbs.amap.com/dev/key/app
> **DashScope 申请**：https://help.aliyun.com/document_detail/2712195.html

## ▶️ 使用方式

### 启动完整导航链路

```bash
# 启动 GPS 驱动（以 ublox 为例）
roslaunch ublox_gps ublox.launch

# 启动 LLM GPS 规划器
rosrun testgps llm_gps_planner.py

# 启动 GPS 导航器
rosrun testgps gps_waypoint_navigator.py
```

### 发送导航指令

```python
# 通过ROS话题或服务发送指令
rostopic pub /navigation_goal std_msgs/String "data: '去西门'"
# 或
rostopic pub /navigation_goal std_msgs/String "data: '图书馆怎么走'"
```

### 坐标转换测试

```python
from coordTransform_utils import gcj02_to_wgs84, wgs84_to_gcj02

# GCJ-02（高德坐标）→ WGS-84（GPS原始坐标）
wgs_lng, wgs_lat = gcj02_to_wgs84(117.283, 31.843)

# WGS-84 → GCJ-02
gcj_lng, gcj_lat = wgs84_to_gcj02(117.276, 31.848)
```

## 📌 坐标系精度说明

| 转换 | 精度 | 说明 |
|------|------|------|
| WGS-84 ↔ GCJ-02 | ±0.5m | 使用官方非线性加密算法近似 |
| GCJ-02 ↔ BD-09 | ±0.1m | 百度公开算法，精度较高 |
| WGS-84 ↔ BD-09 | ±0.5m | 通过 GCJ-02 中转 |

中国法律要求所有公开地图产品使用 GCJ-02 坐标系，GPS 传感器原始输出为 WGS-84，因此在面向高德 API 时必须进行坐标转换。

## 🗺️ ROS 节点图

```
/fix (sensor_msgs/NavSatFix)       GPS 接收机
    │
    ▼
gps_waypoint_navigator ──→ /cmd_vel (geometry_msgs/Twist)
    │
    ▼
/navigation_goal (std_msgs/String)
    │
    ▼
llm_gps_planner ──→ Qwen API ──→ AMap API ──→ /target_point
    │                                                    │
    ▼                                                    ▼
coord_transform_robot ──→ coordTransform_utils    target_point.py
```

## 📂 项目结构

```
testGPS/
├── CMakeLists.txt                  # ROS 编译配置
├── package.xml                     # ROS 包描述
├── README.md                       # 本文件
├── src/                            # C++ 源码目录（可选）
└── scripts/                        # Python 脚本
    ├── llm_gps_planner.py          # LLM 规划器
    ├── gps_waypoint_navigator.py   # GPS 导航器
    ├── coordTransform_utils.py     # 坐标转换工具函数
    ├── coord_transform_robot.py    # 机器人坐标转换
    ├── target_point.py             # 目标点定义
    ├── odom_gps_aligner.py         # 里程计-GPS 对齐
    ├── alignment_motion.py         # 校准运动控制
    ├── hello.py                    # 测试脚本
    └── ustc_gaoxin_poi.json        # 中科大高新校区 POI
```

## ⚙️ 自定义扩展

### 添加新的 POI 数据

编辑 `ustc_gaoxin_poi.json`：

```json
[
  {
    "name": "西门",
    "lng": 117.2735,
    "lat": 31.8428,
    "description": "高新区校区西门"
  }
]
```

### 替换地图服务商

目前使用高德 API，如替换为百度地图：
1. 在 `coordTransform_utils.py` 中启用 BD-09 转换
2. 在 `llm_gps_planner.py` 中更换地图 API 调用

### 更换 LLM

`llm_gps_planner.py` 中可替换为其他大模型（GPT、文心等），只需修改 API 调用接口。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 🙏 致谢

- [高德开放平台](https://lbs.amap.com/) — 地图服务与路径规划
- [阿里云 DashScope / 通义千问](https://dashscope.aliyun.com/) — 大模型能力
- [ROS](https://www.ros.org/) — 机器人操作系统
