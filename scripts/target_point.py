#!/usr/bin/env python3
"""
target_point.py — 局部轨迹点执行控制器（带朝向控制）
=====================================================
功能：
  接收 vla_navigation_node 发布的全局坐标系 (odom) 目标点，
  利用 odometry 反馈闭环控制机器人到达目标，加入 yaw 速度
  控制使机器人尽量朝向目标前进。

通信：
  Sub: /local_waypoint_cmd (PointStamped, frame_id="odom")
       point.x/y = 目标点在 odom 坐标系下的坐标 (米)
  Sub: /odom (Odometry)                          里程计
  Pub: /local_waypoint_reached (Bool)            到达/超时信号
       True  = 正常到达 (距离 < 阈值)
       False = 超时 (30s 未到达，强制跳过)

控制策略：
  1. 计算目标在全局坐标系下的方位角 → heading_error
  2. P 控制器生成 yaw 速度，使机器人朝向目标
  3. 目标在侧后方 (|error| > 85°) → 优先原地旋转
  4. 目标在前方 → 边运动边朝向，转向幅度大时减速
  5. 到达判定：距离 < 阈值 → 停止 → 发布 True
  6. 超时判定：单点超过 30s → 停止 → 发布 False
"""

import rospy
import math
import sys

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

# ── Unitree SDK ──────────────────────────────────────────
sys.path.insert(0, "/home/robot/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
MOVE_SPEED        = 0.60    # 最大运动速度 (m/s)
YAW_KP            = 1.5     # 朝向控制 P 增益
YAW_MAX           = 1.0     # 最大转向角速度 (rad/s)
CTRL_RATE         = 50      # 控制循环频率 (Hz)

REACH_THRESHOLD   = 0.10    # 到达判定距离 (m) — 防"刚到就到达"提前中断
TIMEOUT_SECONDS   = 30.0    # 单点最大执行时间 (s)
STOP_THRESHOLD    = 0.05    # 停止信号判定阈值 (m)
MIN_FORWARD_SPEED = 0.08    # 原地旋转时的最小前向速度 (m/s)


class TargetPointController:
    """
    局部轨迹点执行控制器（带朝向控制）

    接收 odom 坐标系下的目标点，闭环追踪到达。
    """

    def __init__(self, iface: str):
        # ── Unitree SDK ────────────────────────────────────
        ChannelFactoryInitialize(0, iface)
        self.sc = SportClient()
        self.sc.Init()
        rospy.loginfo(f"✅ Unitree SportClient 初始化成功 (iface={iface})")

        # ── ROS 节点 ───────────────────────────────────────
        rospy.init_node("target_point_controller", anonymous=True)
        rospy.Subscriber("/local_waypoint_cmd", PointStamped, self.cmd_callback, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        self.reached_pub = rospy.Publisher("/local_waypoint_reached", Bool, queue_size=1)

        # ── 状态 ───────────────────────────────────────────
        self.current_odom: Odometry = None
        self.world_target_x: float = 0.0   # 目标在 odom 下的 x
        self.world_target_y: float = 0.0   # 目标在 odom 下的 y
        self.is_running: bool = False
        self.start_time: rospy.Time = None

        rospy.loginfo("=" * 50)
        rospy.loginfo(" TargetPointController（带朝向控制）")
        rospy.loginfo(f" speed={MOVE_SPEED}, yaw_kp={YAW_KP}")
        rospy.loginfo(f" reach={REACH_THRESHOLD}m, timeout={TIMEOUT_SECONDS}s")
        rospy.loginfo("=" * 50)

    # ══════════════════════════════════════════════════════
    #  回调函数
    # ══════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry):
        """里程计回调"""
        self.current_odom = msg

    def cmd_callback(self, msg: PointStamped):
        """接收全局坐标系 (odom) 下的目标点"""
        tx, ty = msg.point.x, msg.point.y

        # ── 停止信号 (0,0) ─────────────────────────────
        if abs(tx) < STOP_THRESHOLD and abs(ty) < STOP_THRESHOLD:
            rospy.loginfo("🛑 收到停止信号，停止运动")
            self.sc.StopMove()
            self.is_running = False
            return

        # ── 直接使用世界坐标（已由 vla_navigation 转换） ──
        self.world_target_x = tx
        self.world_target_y = ty
        self.is_running = True
        self.start_time = rospy.Time.now()

        # ── 如果目标过近，直接确认 ──────────────────────
        if self.current_odom is not None:
            cx = self.current_odom.pose.pose.position.x
            cy = self.current_odom.pose.pose.position.y
            d = math.hypot(tx - cx, ty - cy)
            rospy.loginfo(f"📩 全局目标: ({tx:.2f}, {ty:.2f})m, 距离={d:.2f}m")
            if d < REACH_THRESHOLD:
                rospy.loginfo(f"⚡ 目标过近 (d={d:.2f}m)，直接确认到达")
                self.sc.StopMove()
                self.is_running = False
                self.reached_pub.publish(Bool(True))
        else:
            rospy.loginfo(f"📩 全局目标: ({tx:.2f}, {ty:.2f})m")

    # ══════════════════════════════════════════════════════
    #  工具函数
    # ══════════════════════════════════════════════════════

    def _get_yaw(self) -> float:
        """从 odom 提取 yaw (弧度)"""
        q = self.current_odom.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ══════════════════════════════════════════════════════
    #  主控制循环
    # ══════════════════════════════════════════════════════

    def run(self):
        """控制主循环"""
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            if self.is_running:
                self._move_to_target()
            rate.sleep()

    def _move_to_target(self):
        """
        闭环控制：朝向目标运动

        每步：
          1. 从 odom 读取当前位置
          2. 计算与目标的偏差 (dx, dy)
          3. 到达/超时判定
          4. 计算 heading_error → yaw_rate (P 控制)
          5. 分解 body-frame 速度 → Move(vx, vy, vyaw)
        """
        if self.current_odom is None:
            return

        # ── 当前位姿 ──────────────────────────────────────
        cx = self.current_odom.pose.pose.position.x
        cy = self.current_odom.pose.pose.position.y
        yaw = self._get_yaw()

        # ── 世界坐标系下的偏差 ────────────────────────────
        dx = self.world_target_x - cx
        dy = self.world_target_y - cy
        dist = math.hypot(dx, dy)

        # ── (1) 到达判定 ──────────────────────────────────
        if dist < REACH_THRESHOLD:
            rospy.loginfo(f"   ✅ 到达目标 (剩余={dist:.3f}m < {REACH_THRESHOLD}m)")
            self.sc.StopMove()
            self.is_running = False
            self.reached_pub.publish(Bool(True))
            return

        # ── (2) 超时判定 ──────────────────────────────────
        if self.start_time is not None:
            elapsed = (rospy.Time.now() - self.start_time).to_sec()
            if elapsed > TIMEOUT_SECONDS:
                rospy.logwarn(f"   ⚠️ 超时 ({elapsed:.1f}s), 剩余={dist:.2f}m, 强制跳过")
                self.sc.StopMove()
                self.is_running = False
                self.reached_pub.publish(Bool(False))
                return

        # ══════════════════════════════════════════════════
        #  朝向控制 + 运动控制
        # ══════════════════════════════════════════════════
        # 目标在世界坐标系下的方位角
        target_angle = math.atan2(dy, dx)

        # 机器人航向偏差 (归一化到 [-π, π])
        heading_error = target_angle - yaw
        heading_error = math.atan2(math.sin(heading_error),
                                   math.cos(heading_error))

        if abs(heading_error) > math.radians(85):
            # ── 目标在侧后方 → 优先转向 ────────────────
            yaw_rate = math.copysign(YAW_MAX, heading_error)
            vx = MIN_FORWARD_SPEED
            vy = 0.0
        else:
            # ── 目标在前方 → 边运动边朝向 ──────────────
            yaw_rate = YAW_KP * heading_error
            yaw_rate = max(-YAW_MAX, min(YAW_MAX, yaw_rate))

            # 世界偏差 → 机器人 body 坐标系
            body_dx = dx * math.cos(yaw) + dy * math.sin(yaw)
            body_dy = -dx * math.sin(yaw) + dy * math.cos(yaw)

            # 转向幅度大时减速，使转弯更稳
            turn_factor = max(0.5, 1.0 - abs(heading_error) / math.radians(45))
            vx = MOVE_SPEED * (body_dx / dist) * turn_factor
            vy = MOVE_SPEED * (body_dy / dist) * turn_factor

        # ── (5) 发送运动指令 ─────────────────────────────
        try:
            self.sc.Move(vx, vy, yaw_rate)
        except Exception as e:
            rospy.logerr_throttle(5, f"Move 指令异常: {e}")

        # ── 降频日志 ──────────────────────────────────────
        elapsed = (rospy.Time.now() - self.start_time).to_sec() if self.start_time else 0
        if int(elapsed * 2) % 10 == 0:
            rospy.loginfo(f"   🏃 dist={dist:.2f}m, vx={vx:.3f}, vy={vy:.3f}, "
                          f"vyaw={yaw_rate:.3f}, angle_err={math.degrees(heading_error):.1f}°, "
                          f"t={elapsed:.1f}s")

    def shutdown(self):
        """清理"""
        rospy.loginfo("🛑 关闭 TargetPointController，停止机器人")
        try:
            self.sc.StopMove()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 target_point.py <网络接口名>")
        print("示例: python3 target_point.py enp58s0")
        sys.exit(1)

    ctrl = TargetPointController(sys.argv[1])
    try:
        ctrl.run()
    except rospy.ROSInterruptException:
        ctrl.shutdown()
    except Exception as e:
        rospy.logerr(f"❌ TargetPointController 异常: {e}", exc_info=True)
        ctrl.shutdown()
