#!/usr/bin/env python3
"""
alignment_motion.py — 对齐模式运动控制 (testgps)
──────────────────────────────────────────────
功能：
  仿照 target_point.py，但对齐时控制机器人沿初始航向匀速直走，
  通过 Odometry 反馈做闭环航向保持，避免漂移。

通信：
  Sub: /alignment_mode  (std_msgs/Bool)
       True  = 匀速直走（由 odom_gps_aligner 监视 odom 决定何时停）
       False = 停止运动
  Sub: /odom            (nav_msgs/Odometry)  里程计反馈（航向保持用）

启动方式：
  rosrun testgps alignment_motion.py <网络接口名>
  例: rosrun testgps alignment_motion.py enp58s0
"""

import math
import rospy
import sys
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry

# ── Unitree SDK ──────────────────────────────────────────
sys.path.insert(0, "/home/robot/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
MOVE_SPEED = 0.5       # 对齐时前进速度 (m/s)
YAW_KP     = 1.5       # 航向保持 P 增益
YAW_MAX    = 0.5       # 最大纠偏角速度 (rad/s)
CTRL_RATE  = 50        # 控制频率 (Hz)


class AlignmentMotion:
    """
    对齐运动节点（带航向保持）

    由 /alignment_mode 信号控制：
      True  → 沿初始航向匀速直走（odometry 闭环保持方向）
      False → 停止
    """

    def __init__(self, iface: str):
        # ── Unitree SDK ────────────────────────────────────
        ChannelFactoryInitialize(0, iface)
        self.sc = SportClient()
        self.sc.Init()

        rospy.init_node("alignment_motion", anonymous=False)

        # ── 状态 ───────────────────────────────────────────
        self._alignment_mode = False
        self._prev_alignment_mode = False   # 记录上一周期的值，检测边沿
        self._current_odom = None           # 最新里程计
        self._initial_yaw = 0.0             # 启动对齐时的航向（弧度）
        self._initial_yaw_locked = False    # 初始航向是否已记录

        # ── 订阅 ───────────────────────────────────────────
        rospy.Subscriber("/alignment_mode", Bool, self._mode_callback)
        rospy.Subscriber("/odom", Odometry, self._odom_callback)

        rospy.loginfo("[align_motion] 启动, iface=%s, speed=%.2f m/s, yaw_kp=%.1f",
                      iface, MOVE_SPEED, YAW_KP)
        rospy.loginfo("[align_motion] 等待 /alignment_mode 信号...")

    def _mode_callback(self, msg: Bool):
        """对齐模式开关"""
        self._alignment_mode = msg.data
        if msg.data:
            # 对齐开始时，记录当前航向作为目标航向
            if self._current_odom is not None:
                self._initial_yaw = self._get_yaw_from_odom()
                self._initial_yaw_locked = True
                rospy.loginfo("[align_motion] 🔧 对齐模式 ON → 匀速直走 (目标航向=%.1f°)",
                              math.degrees(self._initial_yaw))
            else:
                self._initial_yaw = 0.0
                self._initial_yaw_locked = False
                rospy.loginfo("[align_motion] 🔧 对齐模式 ON → 匀速直走 (无odom，不纠偏)")
        else:
            rospy.loginfo("[align_motion] 🔧 对齐模式 OFF → 停止")

    def _odom_callback(self, msg: Odometry):
        """里程计回调"""
        self._current_odom = msg

    def _get_yaw_from_odom(self) -> float:
        """从 odom 四元数提取 yaw"""
        q = self._current_odom.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def run(self):
        """主循环 — 带航向保持的匀速直走"""
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            cur = self._alignment_mode
            prev = self._prev_alignment_mode
            self._prev_alignment_mode = cur

            if cur and not prev:
                # 上升沿：开始直走（初始航向已在 mode_callback 中记录）
                rospy.loginfo("[align_motion] ▶ 开始前进 (航向保持开启)")

            if cur:
                # ── 带航向保持的直走 ────────────────────────
                yaw_rate = 0.0
                if self._initial_yaw_locked and self._current_odom is not None:
                    current_yaw = self._get_yaw_from_odom()
                    yaw_error = current_yaw - self._initial_yaw
                    yaw_error = math.atan2(math.sin(yaw_error),
                                           math.cos(yaw_error))  # 归一化
                    yaw_rate = -YAW_KP * yaw_error
                    yaw_rate = max(-YAW_MAX, min(YAW_MAX, yaw_rate))

                self.sc.Move(MOVE_SPEED, 0.0, yaw_rate)

            elif prev and not cur:
                # 下降沿：停一次，之后静默让出控制权给 target_point
                rospy.loginfo("[align_motion] ⏹ 停止，让出控制权")
                self._initial_yaw_locked = False
                self.sc.StopMove()
            # else: 两个都是 False → 不发送任何指令，不干扰其他节点

            rate.sleep()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: rosrun testgps alignment_motion.py <网络接口名>")
        print("示例: rosrun testgps alignment_motion.py enp58s0")
        sys.exit(1)

    node = AlignmentMotion(sys.argv[1])
    try:
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[align_motion] 节点退出")
    except Exception as e:
        rospy.logerr("[align_motion] 错误: %s", e)
        import traceback
        traceback.print_exc()
