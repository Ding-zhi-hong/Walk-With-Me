#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CityWalker 导航控制节点

整体流程：
1. 初始化 TimeWindowCache（实时采集图像+里程计）
2. 预加载 CityWalker 模型到 GPU
3. 订阅 /fix、/odom、/planned_path、/yaw_alignment
4. 遍历全局规划路径的途经点，逐个处理：
   a) 截取历史 5 帧图像+相对位姿 (data_collector)
   b) 计算目标点在机器人坐标系下的相对位姿 (coord_transform_robot)
   c) CityWalker 模型推理生成未来 5 个轨迹点（机器人坐标系相对坐标）
   d) 将 5 个轨迹点转换到世界坐标系 (odom)
   e) 逐一发送给 target_point（带朝向控制，世界坐标）
   f) 5 个轨迹点完成后检查是否到达全局途经点
      - 到达 → 切换下一个全局途经点
      - 未到达 → 重新采集数据 + 模型推理，继续导航
5. 全部全局途经点完成 → 导航结束

通信:
  Sub: /fix                            GPS (WGS84)
  Sub: /odom                           轮式里程计
  Sub: /planned_path                   全局规划路径 (GCJ-02)
  Sub: /yaw_alignment                  Odom↔GPS 对齐角度
  Sub: /local_waypoint_reached         局部轨迹点到达确认 (来自 target_point)
  Pub: /local_waypoint_cmd            世界坐标轨迹点 (odom frame) → target_point
  Pub: /citywalker_predicted_path     预测路径可视化 (odom frame)
  Pub: /current_target_waypoint        当前目标全局途经点

依赖：
  - data_collector.TimeWindowCache (实时数据窗口)
  - loadmodel.CityWalkerInferencer (模型推理)
  - coord_transform_robot.get_target_in_robot_frame_from_odom (坐标转换)
  - target_point.TargetPointController (底层运动控制，通过 ROS 话题通信)
"""
import rospy
import math
import numpy as np
import os
import sys
from typing import List, Tuple, Optional

# ROS 消息
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool, Float64, Header

# 动态路径导入
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 自定义模块
from data_collector import TimeWindowCache
from loadmodel import CityWalkerInferencer
from coord_transform_robot import get_target_in_robot_frame_from_odom


class CityWalkerNavigationNode:
    """
    CityWalker 导航节点（主体）

    通过 CityWalker 模型逐段生成局部轨迹，控制机器人沿全局规划路径行驶。
    """

    def __init__(self):
        rospy.init_node("citywalker_navigation_node", anonymous=True)

        # ==================== 参数配置 ====================
        # 全局途经点到达阈值（米）—— 判断机器人是否到达 planned_path 上的点
        self.reach_threshold_global = 2.0
        # 局部轨迹点到达阈值（米）—— 机器人到达 CityWalker 预测点的判定
        self.reach_threshold_local = 0.3
        # 模型推理步长缩放因子（由 CityWalker 训练决定）
        self.step_scale = 0.60
        # 连续失败最大重试次数（同一全局点生成局部点失败计数）
        self.max_retries_per_waypoint = 5
        # 每次 CityWalker 推理后，走完多少个局部点就重新生成（避免偏差累积）
        # 设为5表示走完全部5个预测点再重新生成，让机器人有足够的运动积累
        self.waypoints_per_step = 5
        # 首次导航前是否已完成初始微动（前进0.5m打破视觉僵局）【已废弃，改用种子历史】
        self._initial_warmup_done: bool = True  # 跳过物理微动
        # Odom ↔ GPS(UTM) 对齐角度（由 odom_gps_aligner 发布，初始化时为0）
        self.yaw_correction: float = 0.0
        self.yaw_align_received: bool = False

        # ==================== 时间窗口缓存（数据采集层） ====================
        # 采集最近 10 秒的图像 + 里程计数据，实时刷新
        rospy.loginfo("🔄 初始化 TimeWindowCache（时间窗口缓存）...")
        self.time_cache = TimeWindowCache(
            window_seconds=10.0,
            out_img_w=630,
            out_img_h=350
        )
        # 等待缓存积累足够数据（建议 ≥ 5 秒）
        rospy.sleep(10)
        rospy.loginfo("✅ TimeWindowCache 初始化完成")

        # 等待 Odom↔GPS 对齐角度（来自 odom_gps_aligner，非阻塞）
        if not self.yaw_align_received:
            rospy.loginfo("🌐 等待 Odom↔GPS 对齐角度 (/yaw_alignment) ... "
                          f"(当前 correction={self.yaw_correction:.2f}°)")
            # 非阻塞等待：以下主循环中会持续接收；此处仅在日志提示
            # 对齐期间 yaw_correction=0，导航逻辑仍然可运行

        # ==================== 预加载模型到 GPU ====================
        rospy.loginfo("🔄 加载 CityWalker 模型到 GPU...")
        self.model = CityWalkerInferencer()
        rospy.loginfo("✅ CityWalker 模型加载完成")

        # ==================== 全局状态变量 ====================
        # 传感器数据
        self.current_gps: Optional[NavSatFix] = None
        self.current_odom: Optional[Odometry] = None
        # 全局规划路径
        self.planned_path: Optional[Path] = None
        # 当前处理的全局途经点索引（0-based）
        self.global_wp_index: int = 0
        # 当前轮次生成的局部轨迹点列表（机器人坐标系下的相对坐标）
        self.local_waypoints: List[Tuple[float, float]] = []
        # 当前正在执行的局部轨迹点索引
        self.local_wp_index: int = 0
        # 是否正在等待局部点到达确认
        self.waiting_local_reach: bool = False
        # 当前局部点的 retry 计数（防止无限循环）
        self.retry_count: int = 0
        # 本轮是否已完成所有全局途经点
        self.all_waypoints_done: bool = False

        # ==================== ROS 订阅器 ====================
        rospy.Subscriber("/fix", NavSatFix, self.gps_callback, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber("/planned_path", Path, self.path_callback, queue_size=1)
        # 接收 odom_gps_aligner 的对齐角度（Odom ↔ GPS/UTM 方向对齐）
        rospy.Subscriber("/yaw_alignment", Float64, self.yaw_align_callback, queue_size=1)
        # 接收 target_point 的局部点到达确认
        rospy.Subscriber("/local_waypoint_reached", Bool, self.local_reached_callback, queue_size=1)

        # ==================== ROS 发布器 ====================
        # 发送局部轨迹点给 target_point 执行
        self.local_wp_pub = rospy.Publisher(
            "/local_waypoint_cmd", PointStamped, queue_size=1
        )
        # 发布 CityWalker 预测路径（可视化/调试用）
        self.predicted_path_pub = rospy.Publisher(
            "/citywalker_predicted_path", Path, queue_size=10
        )
        # 发布当前目标全局途经点（可视化/调试用）
        self.target_wp_pub = rospy.Publisher(
            "/current_target_waypoint", PoseStamped, queue_size=1
        )

        rospy.loginfo("✅ CityWalker 导航节点启动完成，等待 `/planned_path` 规划路径...")
        rospy.loginfo("🌐 订阅 /yaw_alignment (Odom↔GPS对齐)，初始 correction=0.0°")

    # ====================================================================
    # ROS 回调函数
    # ====================================================================

    def gps_callback(self, msg: NavSatFix):
        """GPS 定位回调"""
        if not np.isnan(msg.latitude) and not np.isnan(msg.longitude):
            self.current_gps = msg

    def odom_callback(self, msg: Odometry):
        """里程计回调"""
        self.current_odom = msg

    def yaw_align_callback(self, msg: Float64):
        """Odom↔GPS 对齐角度回调（来自 odom_gps_aligner）"""
        self.yaw_correction = msg.data
        if not self.yaw_align_received:
            self.yaw_align_received = True
            rospy.loginfo(f"🌐 收到 yaw 对齐角度: {math.degrees(self.yaw_correction):.2f}°")

    def path_callback(self, msg: Path):
        """
        全局规划路径回调（由 llm_gps_planner.py 发布 /planned_path）

        planned_path 的每个 pose 中:
          - pose.position.x = 目标点经度 (GCJ-02, 对应 longitude)
          - pose.position.y = 目标点纬度 (GCJ-02, 对应 latitude)
        """
        if not msg.poses:
            rospy.logwarn("⚠️ 收到空的全局规划路径")
            return

        self.planned_path = msg
        self.global_wp_index = 0
        self.local_waypoints.clear()
        self.local_wp_index = 0
        self.waiting_local_reach = False
        self.retry_count = 0
        self.all_waypoints_done = False
        self._stop_robot()  # 重置时停住机器人，等待新轨迹生成

        rospy.loginfo(f"📦 收到规划路径，共 {len(msg.poses)} 个全局途经点")
        rospy.loginfo(f"   🎯 首点: lon={msg.poses[0].pose.position.x:.6f}, lat={msg.poses[0].pose.position.y:.6f}")
        rospy.loginfo(f"   🏁 终点: lon={msg.poses[-1].pose.position.x:.6f}, lat={msg.poses[-1].pose.position.y:.6f}")

    def local_reached_callback(self, msg: Bool):
        """
        局部轨迹点到达确认回调（来自 target_point.py）

        当 target_point 确认到达一个局部轨迹点（或超时）时回调。
        """
        # 只处理 True（到达），False 由超时逻辑产生，记录警告后同样推进
        if self.waiting_local_reach:
            if msg.data:
                rospy.loginfo(f"   ✅ 局部轨迹点 [{self.local_wp_index + 1}/{len(self.local_waypoints)}] 到达")
            else:
                rospy.logwarn(f"   ⚠️ 局部轨迹点 [{self.local_wp_index + 1}/{len(self.local_waypoints)}] 超时未达，强制推进")
            # 前进到下一个局部点
            self.local_wp_index += 1
            self.waiting_local_reach = False

    # ====================================================================
    # 核心功能函数
    # ====================================================================

    def reached_global_waypoint(self, target_pose_stamped: PoseStamped) -> Tuple[bool, float]:
        """
        使用坐标变换检查是否到达全局途经点

        参数:
            target_pose_stamped: 目标全局途经点 PoseStamped
                (position.x=经度GCJ-02, position.y=纬度GCJ-02)

        返回:
            (是否到达, 距离米)
        """
        if self.current_odom is None or self.current_gps is None:
            return False, float('inf')

        target_lon_gcj = target_pose_stamped.pose.position.x  # 经度
        target_lat_gcj = target_pose_stamped.pose.position.y  # 纬度

        _, _, distance, reached = get_target_in_robot_frame_from_odom(
            robot_lat_wgs=self.current_gps.latitude,
            robot_lon_wgs=self.current_gps.longitude,
            odom_msg=self.current_odom,
            target_lat_gcj=target_lat_gcj,
            target_lon_gcj=target_lon_gcj,
            reach_threshold=self.reach_threshold_global,
            yaw_correction=self.yaw_correction
        )
        return reached, distance

    def generate_local_waypoints(self, target_pose_stamped: PoseStamped) -> bool:
        """
        通过 CityWalker 模型生成局部轨迹点

        流程:
          1. 从 TimeWindowCache 采集最近 5 帧（图像 + 相对位姿）
          2. 计算目标全局点在机器人坐标系下的相对位姿
          3. 模型推理 → 5 个未来轨迹点（机器人坐标系）
          4. 发布预测路径供可视化

        参数:
            target_pose_stamped: 当前目标全局途经点

        返回:
            bool: 是否成功生成局部点
        """
        # ════════════════════════════════════════════════════════════
        #  [已废弃] 初始微动：不再物理前进，改用种子历史
        # ════════════════════════════════════════════════════════════

        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"  🤖 CityWalker 推理 [全局点 {self.global_wp_index + 1}/{len(self.planned_path.poses)}]")
        rospy.loginfo(f"{'='*60}")

        # ---- (1) 从时间窗口缓存采集数据 ----
        citywalker_input = self.time_cache.get_citywalker_input()
        if citywalker_input is None:
            rospy.logwarn("⚠️ 时间窗口缓存数据不足，跳过本次推理")
            return False

        past_poses, processed_imgs = citywalker_input
        rospy.loginfo(f"📸 采集到 {len(processed_imgs)} 帧图像 + {len(past_poses)} 个历史相对位姿")

        # ---- (2) 计算目标点在机器人坐标系下的相对位姿 ----
        target_lon_gcj = target_pose_stamped.pose.position.x  # 经度
        target_lat_gcj = target_pose_stamped.pose.position.y  # 纬度

        x_forward, y_left, distance, _ = get_target_in_robot_frame_from_odom(
            robot_lat_wgs=self.current_gps.latitude,
            robot_lon_wgs=self.current_gps.longitude,
            odom_msg=self.current_odom,
            target_lat_gcj=target_lat_gcj,
            target_lon_gcj=target_lon_gcj,
            reach_threshold=999.0,  # 禁用到达判定（仅用于获取坐标）
            yaw_correction=self.yaw_correction
        )
        target_pose = [x_forward, y_left]
        rospy.loginfo(f"📐 目标点相对位姿: 前向={x_forward:.2f}m, 左侧={y_left:.2f}m, 直线距离={distance:.2f}m")

        # ---- (3) 模型推理 ----
        # past_poses 格式: [(x_T-4, y_T-4), ..., (x_T0, y_T0)]
        # 模型要求: 4 个历史位姿 + 自动补 [0,0] 作为当前
        history_poses = list(past_poses[:4])  # copy
        rospy.logdebug(f"📥 历史位姿: {history_poses}")
        rospy.logdebug(f"📥 目标位姿: {target_pose}")

        # 种子历史：历史接近零时在本体前向生成虚拟运动（过去位置在后方）
        # 不再指向目标方向，因为机器人可能朝向任意方向
        # 模型训练数据来自步行/驾驶视频，从未见过全零历史输入
        # 没有运动历史 → 模型输出≈0 → 死锁
        min_history_motion = 0.05  # 5cm
        history_magnitude = max(abs(p[0]) + abs(p[1]) for p in history_poses)
        if history_magnitude < min_history_motion:
            step = self.step_scale / 2.0  # 0.30m
            for i in range(4):
                t = (4 - i) * step       # 1.2, 0.9, 0.6, 0.3
                history_poses[i] = (-t, 0.0)  # (forward=-t, left=0)
            rospy.loginfo(f"🌱 历史接近零(mag={history_magnitude:.3f}), "
                         f"种子本体前向历史(step={step:.2f}m)")
        else:
            rospy.loginfo(f"📊 历史运动幅度: {history_magnitude:.3f}m")

        result = self.model.predict(
            images=processed_imgs,
            history_poses=history_poses,
            target_pose=target_pose,
            save_results=False
        )

        # ---- (4) 后处理 ----
        # result["waypoints"] 已由模型内部乘以 step_scale，单位为米
        predicted_local_waypoints = result["waypoints"]  # shape: (5, 2)
        arrive_confidence = result["arrival_confidence"]

        # ---- (5) 发布预测路径用于可视化 ----
        self.publish_predicted_path(predicted_local_waypoints)

        # ---- (6) 转换到世界坐标系 (odom) ----
        # 注意：用纯 odom_yaw 做 body→odom 转换，不需要 yaw_correction
        # yaw_correction 是 odom↔GPS(ENU) 之间的夹角，仅在 GPS 计算中用
        robot_x = self.current_odom.pose.pose.position.x
        robot_y = self.current_odom.pose.pose.position.y
        robot_yaw = self._get_yaw_from_odom()

        predicted_list = predicted_local_waypoints.tolist()
        self.local_waypoints = []
        for lx, ly in predicted_list:
            wx = robot_x + lx * math.cos(robot_yaw) - ly * math.sin(robot_yaw)
            wy = robot_y + lx * math.sin(robot_yaw) + ly * math.cos(robot_yaw)
            self.local_waypoints.append((wx, wy))

        self.local_wp_index = 0
        self.waiting_local_reach = False

        rospy.loginfo(f"📍 CityWalker 生成 {len(self.local_waypoints)} 个局部轨迹点 (odom 坐标系), 到达置信度={arrive_confidence:.3f}")
        for i, (wx, wy) in enumerate(self.local_waypoints):
            rospy.loginfo(f"   t+{i + 1}: world=({wx:.3f}, {wy:.3f})m")
        rospy.loginfo("")

        return True

    def send_local_waypoint(self, x: float, y: float):
        """
        发送一个全局轨迹点给 target_point 控制器执行

        参数:
            x: 目标点在 odom 坐标系下的 x (米)
            y: 目标点在 odom 坐标系下的 y (米)
        """
        msg = PointStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0

        self.local_wp_pub.publish(msg)
        rospy.loginfo(f"   🚀 发送全局轨迹点 [{self.local_wp_index + 1}/{len(self.local_waypoints)}]: "
                       f"world=({x:.3f}, {y:.3f})m")

    def _send_warmup_waypoint(self, x: float, y: float):
        """
        发送初始微动目标点（专用于 warmup，不依赖 local_waypoints 列表）

        参数:
            x: 目标点在 odom 坐标系下的 x (米)
            y: 目标点在 odom 坐标系下的 y (米)
        """
        msg = PointStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.local_wp_pub.publish(msg)
        rospy.loginfo(f"   🚀 [微动] 目标: world=({x:.3f}, {y:.3f})m")

    def publish_predicted_path(self, waypoints: np.ndarray):
        """
        发布 CityWalker 预测路径（可视化用）

        将机器人坐标系下的局部点转换为 odom 坐标系下的全局路径。
        """
        if self.current_odom is None:
            return

        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")

        robot_x = self.current_odom.pose.pose.position.x
        robot_y = self.current_odom.pose.pose.position.y
        # 可视化用纯 odom_yaw（与上面 trajectory 转换保持一致）
        robot_yaw = self._get_yaw_from_odom()

        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)

        for i, (lx, ly) in enumerate(waypoints):
            # 将机器人坐标系 (lx=前向, ly=左侧) 旋转到 odom 坐标系
            world_x = robot_x + lx * cos_y - ly * sin_y
            world_y = robot_y + lx * sin_y + ly * cos_y

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.predicted_path_pub.publish(path_msg)

    def _get_yaw_from_odom(self) -> float:
        """从 odom 消息提取 yaw 角"""
        q = self.current_odom.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ====================================================================
    # 导航主循环
    # ====================================================================

    def navigation_step(self):
        """
        单步导航逻辑（由主循环按频率调用）

        状态机:
          S0: 等待数据就绪
          S1: 检查当前全局途经点是否到达
               到达 → 推进到下一个全局点，清空局部点
          S2: 检查是否需重新生成局部点
               局部点用尽 or 首次生成 → 调用模型生成
          S3: 发送下一个局部轨迹点给 target_point
          S4: 等待 target_point 到达确认（在回调中处理）
        """
        # ====== S0: 数据就绪检查 ======
        if self.planned_path is None:
            return  # 还未收到规划路径
        if self.current_gps is None or self.current_odom is None:
            return  # 传感器数据未就绪
        if self.all_waypoints_done:
            return  # 全部完成

        # ====== S1: 检查当前全局途经点 ======
        if self.global_wp_index >= len(self.planned_path.poses):
            # 所有全局点已处理完毕
            rospy.loginfo("\n" + "=" * 60)
            rospy.loginfo("  🏁🏁🏁 所有全局途经点导航完成！到达最终目标点 🏁🏁🏁")
            rospy.loginfo("=" * 60)
            self.all_waypoints_done = True
            self.local_waypoints.clear()
            self._stop_robot()
            return

        target_global_wp = self.planned_path.poses[self.global_wp_index]

        # ---- 发布当前目标点（可视化） ----
        self.target_wp_pub.publish(target_global_wp)

        # ---- 到达判断（使用正确的地球坐标转换） ----
        reached, distance = self.reached_global_waypoint(target_global_wp)

        if reached:
            rospy.loginfo(f"\n✅✅✅ 到达全局途经点 [{self.global_wp_index + 1}/{len(self.planned_path.poses)}] "
                           f"(距离={distance:.2f}m <= 阈值={self.reach_threshold_global}m)")
            # 先关回调闸门，再清数据（防止竞态）
            self.waiting_local_reach = False
            self.global_wp_index += 1
            self.local_waypoints.clear()
            self.local_wp_index = 0
            self.retry_count = 0

            # 停止机器人运动（重要！确保到达后停在目标点）
            self._stop_robot()
            return

        # ====== S2: 检查是否需要生成局部轨迹点 ======
        # 条件：
        #   - 列表为空（首次生成）
        #   - 索引越界（全部走完）
        #   - 已走满 waypoints_per_step 个点（提前重新生成，避免偏差累积）
        max_local_idx = min(self.waypoints_per_step, len(self.local_waypoints)) if self.local_waypoints else 0
        if (not self.local_waypoints or
                self.local_wp_index >= len(self.local_waypoints) or
                self.local_wp_index >= max_local_idx):

            # 检查重试次数限制
            if self.retry_count >= self.max_retries_per_waypoint:
                rospy.logwarn(f"⚠️ 当前全局点 [{self.global_wp_index + 1}] 连续失败 "
                              f"{self.max_retries_per_waypoint} 次，强制推进到下一个")
                self.waiting_local_reach = False
                self.global_wp_index += 1
                self.local_waypoints.clear()
                self.local_wp_index = 0
                self.retry_count = 0
                return

            # 生成新的局部轨迹点
            rospy.loginfo(f"🔄 正在接近全局点 [{self.global_wp_index + 1}], "
                          f"当前距离={distance:.2f}m, retry={self.retry_count + 1}")
            success = self.generate_local_waypoints(target_global_wp)

            if not success:
                self.retry_count += 1
                return  # 等待下一周期重试

            self.retry_count = 0  # 生成成功，重置计数

        # ====== S3: 发送下一个局部轨迹点 ======
        if not self.waiting_local_reach and self.local_wp_index < len(self.local_waypoints):
            lx, ly = self.local_waypoints[self.local_wp_index]
            self.send_local_waypoint(lx, ly)
            self.waiting_local_reach = True

        # ====== S4: 等待 target_point 到达确认 ======
        # 在 local_reached_callback 中处理

    def _stop_robot(self):
        """发布停止信号给 target_point"""
        stop_msg = PointStamped()
        stop_msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        stop_msg.point.x = 0.0
        stop_msg.point.y = 0.0
        stop_msg.point.z = 0.0
        self.local_wp_pub.publish(stop_msg)

    # ====================================================================
    # 主循环
    # ====================================================================

    def run(self):
        """节点主循环，按 10Hz 频率执行导航步进"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.navigation_step()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = CityWalkerNavigationNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("CityWalker 导航节点被中断")
    except Exception as e:
        rospy.logerr(f"❌ CityWalker 导航节点异常: {e}", exc_info=True)