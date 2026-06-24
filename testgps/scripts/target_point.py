#!/usr/bin/env python3
"""
target_point.py — 底层运动控制层（带朝向控制） (testgps)
──────────────────────────────────────────────────────
功能：
  接收 gps_waypoint_navigator 发布的相对目标位姿 (x, y 米，机器人坐标系)，
  驱动机器人向目标运动，加入 yaw 速度控制使机器人尽量朝向目标，
  到达后通知上层。

通信：
  Sub: /local_waypoint_cmd   (geometry_msgs/PointStamped)
       x = 前向距离(m), y = 左侧距离(m) >0
  Pub: /local_waypoint_reached  (std_msgs/Bool)
       True = 当前目标点已到达

  注：x=0, y=0 视为急停指令，立刻停止运动并等待下一个非零指令。

控制策略：
  1. 根据目标相对位置 (tx, ty) 计算朝向角 target_angle = atan2(ty, tx)
  2. 使用 P 控制器生成 yaw 速度：vyaw = YAW_KP * target_angle
  3. 当目标在正前方 (|angle| < 85°)：前后运动 + yaw 调节
  4. 当目标在侧后方 (|angle| ≥ 85°)：优先原地旋转朝向目标
  5. 到达判定：距离 < ARRIVED_THRESHOLD → 停止序列 → 发布 reached

启动方式：
  rosrun testgps target_point.py <网络接口名>
  例: rosrun testgps target_point.py enp58s0
"""

import rospy
import math
import sys

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

# ── Unitree SDK ──────────────────────────────────────────
sys.path.insert(0, "/home/robot/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
MOVE_SPEED           = 0.35   # 最大运动速度 (m/s)
YAW_KP               = 1.5    # 朝向控制 P 增益
YAW_MAX              = 1.0    # 最大转向角速度 (rad/s)
CTRL_RATE            = 50     # 控制循环频率 (Hz)

ARRIVED_THRESHOLD    = 0.30   # 到达判定距离 (m)
ARRIVED_HYSTERESIS   = 0.05   # 迟滞带
STOP_DURATION        = 0.5    # 到达后停止等待时间 (s)
WAYPOINT_TIMEOUT     = 120.0  # 单个目标点超时 (s)

MIN_FORWARD_SPEED    = 0.08   # 原地旋转时的最小前向速度 (m/s)


class TargetPointController:
    """
    底层运动控制器（带朝向控制）

    状态:
      - 接收到新目标 → 运动
      - 到达阈值    → 停止序列 → 发布 reached
      - 超时        → 停止序列
      - 急停指令    → 立即停止
    """

    def __init__(self, iface: str):
        # ── Unitree SDK ────────────────────────────────────
        ChannelFactoryInitialize(0, iface)
        self.sc = SportClient()
        self.sc.Init()

        rospy.init_node("target_point_controller", anonymous=False)

        # ── 发布 / 订阅 ────────────────────────────────────
        rospy.Subscriber(
            "/local_waypoint_cmd", PointStamped, self._cmd_callback
        )
        self._reached_pub = rospy.Publisher(
            "/local_waypoint_reached", Bool, queue_size=1
        )

        # ── 状态 ──────────────────────────────────────────
        self._target_x = 0.0            # 当前目标 x (前向, m)
        self._target_y = 0.0            # 当前目标 y (左侧, m)
        self._target_valid = False      # 是否有有效目标
        self._target_set_time = rospy.Time(0)
        self._stopping = False          # 正在执行停止序列
        self._stop_start_time = rospy.Time(0)
        self._arrived = False           # 已到达标志
        self._was_arrived = False       # 迟滞标志

        rospy.loginfo("[target_point] 运动控制层启动, iface=%s", iface)
        rospy.loginfo("[target_point] speed=%.2f, yaw_kp=%.1f, yaw_max=%.1f",
                      MOVE_SPEED, YAW_KP, YAW_MAX)
        rospy.loginfo("[target_point] arrive=%.2fm, hysteresis=%.2fm",
                      ARRIVED_THRESHOLD, ARRIVED_HYSTERESIS)

    # ══════════════════════════════════════════════════════
    #  回调
    # ══════════════════════════════════════════════════════

    def _cmd_callback(self, msg: PointStamped):
        """收到新的目标点指令"""
        vx = msg.point.x
        vy = msg.point.y

        # ── 急停指令 (0,0) 永远生效 ──────────────────────
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            rospy.loginfo("[target_point] ⛔ 收到停止指令")
            self.sc.StopMove()
            self._target_valid = False
            self._stopping = False
            self._arrived = False
            self._was_arrived = False
            return

        # ── 如果在停止序列中，忽略新指令（防止导航层5Hz
        #     指令不断重置停止序列，导致永远发不出 reached）──
        if self._stopping:
            return

        self._target_x = vx
        self._target_y = vy
        self._target_valid = True
        self._target_set_time = rospy.Time.now()
        self._stopping = False
        self._arrived = False
        self._was_arrived = False

        dist = math.hypot(vx, vy)
        rospy.loginfo("[target_point] 🎯 新目标: rel=(%.2f, %.2f)m, 距离=%.2fm",
                      vx, vy, dist)

    # ══════════════════════════════════════════════════════
    #  运动控制
    # ══════════════════════════════════════════════════════

    def _move_to_target(self):
        """向当前目标运动一步（带朝向控制）"""
        tx = self._target_x
        ty = self._target_y
        dist = math.hypot(tx, ty)
        now = rospy.Time.now()

        # ── 1. 停止序列 ──────────────────────────────────
        if self._stopping:
            if (now - self._stop_start_time).to_sec() >= STOP_DURATION:
                self.sc.StopMove()
                self._target_valid = False
                self._stopping = False
                self._arrived = True
                self._reached_pub.publish(Bool(True))
                rospy.loginfo("[target_point] ✅ 已到达，发送 reached 信号")
            return

        # ── 2. 到达判定（带迟滞） ────────────────────────
        if self._was_arrived:
            if dist < ARRIVED_THRESHOLD + ARRIVED_HYSTERESIS:
                self.sc.StopMove()
                return
            else:
                self._was_arrived = False
                rospy.loginfo("[target_point] 🔄 重新激活, 距离=%.2f > %.2f",
                              dist, ARRIVED_THRESHOLD + ARRIVED_HYSTERESIS)

        if dist < ARRIVED_THRESHOLD:
            rospy.loginfo("[target_point] ✅ 到达阈值 (%.3f < %.2f), "
                          "进入停止序列", dist, ARRIVED_THRESHOLD)
            self._was_arrived = True
            self._stopping = True
            self._stop_start_time = now
            self.sc.StopMove()
            return

        # ── 3. 超时检测 ──────────────────────────────────
        elapsed = (now - self._target_set_time).to_sec()
        if elapsed > WAYPOINT_TIMEOUT:
            rospy.logwarn("[target_point] ⏰ 超时 (%.1f > %.1fs), "
                          "强制跳过", elapsed, WAYPOINT_TIMEOUT)
            self._was_arrived = True
            self._stopping = True
            self._stop_start_time = now
            self.sc.StopMove()
            return

        # ══════════════════════════════════════════════════
        #  4. 运动控制（带 yaw 朝向控制）
        # ══════════════════════════════════════════════════
        # 目标在机器人坐标系下的朝向角
        target_angle = math.atan2(ty, tx)

        # 判断目标是否在后方 (角度 > 85°)
        if abs(target_angle) > math.radians(85):
            # ── 目标在侧后方 → 优先旋转朝向目标 ────────
            yaw_rate = math.copysign(YAW_MAX, target_angle)
            forward_speed = MIN_FORWARD_SPEED
            lateral_speed = 0.0

            rospy.logdebug_throttle(
                1.0, "[target_point] 🔄 转向: angle=%.1f°, vyaw=%.2f",
                math.degrees(target_angle), yaw_rate
            )
        else:
            # ── 目标在前方 → 边运动边朝向目标 ──────────
            # Yaw P 控制：误差角越大，转向越快
            yaw_rate = YAW_KP * target_angle
            yaw_rate = max(-YAW_MAX, min(YAW_MAX, yaw_rate))

            # 前向/侧向速度：按照到目标的方向分解
            # 同时当转向幅度大时略微减速以稳定
            turn_factor = max(0.5, 1.0 - abs(target_angle) / math.radians(45))
            forward_speed = MOVE_SPEED * (tx / dist) * turn_factor
            lateral_speed = MOVE_SPEED * (ty / dist) * turn_factor

        # 发送运动指令
        self.sc.Move(forward_speed, lateral_speed, yaw_rate)

        # 降频日志
        if int(now.to_sec()) % 2 == 0:
            rospy.logdebug_throttle(
                2.0, "[target_point] v=(%.3f, %.3f, %.3f) | dist=%.2f | "
                "angle=%.1f°",
                forward_speed, lateral_speed, yaw_rate,
                dist, math.degrees(target_angle)
            )

    # ══════════════════════════════════════════════════════
    #  主循环
    # ══════════════════════════════════════════════════════

    def run(self):
        """主控制循环"""
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            if self._target_valid:
                self._move_to_target()
            rate.sleep()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: rosrun testgps target_point.py <网络接口名>")
        print("示例: rosrun testgps target_point.py enp58s0")
        sys.exit(1)

    controller = TargetPointController(sys.argv[1])
    rospy.loginfo("[target_point] 启动运动控制循环")
    try:
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[target_point] 节点退出")
    except Exception as e:
        rospy.logerr("[target_point] 错误: %s", e)
        import traceback
        traceback.print_exc()
