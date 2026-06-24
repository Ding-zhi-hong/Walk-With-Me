#!/usr/bin/env python3
"""
alignment_motion.py — 对齐模式运动控制 (testgps)
──────────────────────────────────────────────
功能：
  仿照 target_point.py，但对齐时只需控制机器人匀速直走，
  不进行到达判定和目标跟踪。

通信：
  Sub: /alignment_mode  (std_msgs/Bool)
       True  = 匀速直走（由 odom_gps_aligner 监视 odom 决定何时停）
       False = 停止运动

启动方式：
  rosrun testgps alignment_motion.py <网络接口名>
  例: rosrun testgps alignment_motion.py enp58s0
"""

import rospy
import sys
from std_msgs.msg import Bool

# ── Unitree SDK ──────────────────────────────────────────
sys.path.insert(0, "/home/robot/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
MOVE_SPEED = 0.5       # 对齐时前进速度 (m/s)
CTRL_RATE  = 50        # 控制频率 (Hz)


class AlignmentMotion:
    """
    对齐运动节点

    由 /alignment_mode 信号控制：
      True  → 向前匀速直走
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

        # ── 订阅 ───────────────────────────────────────────
        rospy.Subscriber("/alignment_mode", Bool, self._mode_callback)

        rospy.loginfo("[align_motion] 启动, iface=%s, speed=%.2f m/s",
                      iface, MOVE_SPEED)
        rospy.loginfo("[align_motion] 等待 /alignment_mode 信号...")

    def _mode_callback(self, msg: Bool):
        """对齐模式开关"""
        self._alignment_mode = msg.data
        if msg.data:
            rospy.loginfo("[align_motion] 🔧 对齐模式 ON → 匀速直走")
        else:
            rospy.loginfo("[align_motion] 🔧 对齐模式 OFF → 停止")

    def run(self):
        """主循环"""
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            cur = self._alignment_mode
            prev = self._prev_alignment_mode
            self._prev_alignment_mode = cur

            if cur and not prev:
                # 上升沿：开始直走
                rospy.loginfo("[align_motion] ▶ 开始前进")
            if cur:
                self.sc.Move(MOVE_SPEED, 0.0, 0.0)
            elif prev and not cur:
                # 下降沿：停一次，之后静默让出控制权给 target_point
                rospy.loginfo("[align_motion] ⏹ 停止，让出控制权")
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
