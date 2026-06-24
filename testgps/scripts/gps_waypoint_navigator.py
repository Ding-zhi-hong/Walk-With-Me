#!/usr/bin/env python3
"""
gps_waypoint_navigator.py — GPS 路径逐点导航器 (testgps)
────────────────────────────────────────────────────────
功能：
  1. 接收 /yaw_alignment 对齐角度（来自 odom_gps_aligner）
  2. 等待 llm_gps_planner 的 /planned_path（GCJ-02 路径点）
  3. 对每个路径点：
     a. 结合当前 GPS(WGS84) + 目标点(GCJ-02) + odom 朝向 + 对齐角度
     b. 计算目标点在机器人坐标系下的相对位姿 (x_forward, y_left)
     c. 以固定频率发布给运动层 /local_waypoint_cmd
     d. 通过 GPS 距离 + 运动层 reached 信号双重判定到达
  4. 到达后切换到下一路径点，最终完成导航

数据流：
  llm_gps_planner → /planned_path ─┐
  odom_gps_aligner → /yaw_alignment ─┤
  /fix  (GPS, WGS84) ──────────────┤── gps_waypoint_navigator ── /local_waypoint_cmd ──→ target_point
  /odom (里程计/朝向) ───────────────┤
  target_point → /local_waypoint_reached ─┘
"""

import rospy
import math
import sys
import os

from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float64
from nav_msgs.msg import Odometry

# ── 动态导入坐标转换模块 ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from coord_transform_robot import (
        get_target_in_robot_frame,
        get_yaw_from_odom,
        TRANSFORMER,
    )
except ImportError as e:
    print(f"[gps_navigator] 无法导入 coord_transform_robot: {e}")
    print(f"[gps_navigator] 请确保 coord_transform_robot.py 在: {SCRIPT_DIR}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
GPS_REACH_THRESHOLD   = 1.0     # GPS 距离到达阈值（米）
CTRL_CMD_RATE         = 5       # 向运动层发送指令的频率 (Hz)
WAYPOINT_TIMEOUT      = 120.0   # 单个路径点超时 (s)
SENSOR_WAIT_TIMEOUT   = 10.0    # 传感器等待超时 (s)
ALIGN_WAIT_TIMEOUT    = 30.0    # 对齐角度等待超时 (s)


class GPSWaypointNavigator:
    """
    GPS 路径逐点导航器（纯导航控制层）

    状态机:
      IDLE → PATH_READY → FOLLOWING ─→ TRANSITION ─→ FOLLOWING ─→ … ─→ COMPLETE
                              ↑            │
                              └── 超时/到达 ─┘
    """

    # ── 状态常量 ──────────────────────────────────────────
    STATE_IDLE        = "IDLE"
    STATE_PATH_READY  = "PATH_READY"
    STATE_FOLLOWING   = "FOLLOWING"
    STATE_TRANSITION  = "TRANSITION"
    STATE_COMPLETE    = "COMPLETE"

    def __init__(self):
        rospy.init_node("gps_waypoint_navigator", anonymous=False)

        # ── 状态 ──────────────────────────────────────────
        self._state = self.STATE_IDLE
        self._waypoints = []              # [(lat_gcj, lon_gcj), ...]
        self._current_wp_idx = 0
        self._wp_start_time = rospy.Time(0)
        self._last_cmd_time = rospy.Time(0)

        # ── 传感器缓存 ────────────────────────────────────
        self._current_gps = None          # (lat_wgs84, lon_wgs84)
        self._current_odom = None

        # ── 对齐角度 ──────────────────────────────────────
        self._yaw_correction = 0.0
        self._yaw_align_received = False

        # ── 到达信号（来自 target_point） ─────────────────
        self._reached_signal = False
        self._reached_signal_time = rospy.Time(0)  # 信号到达时间，防跨航点串扰

        # ── GPS 到达迟滞 ──────────────────────────────────
        self._gps_arrived_flag = False   # 上一周期是否已抵达

        # ── 发布器 ────────────────────────────────────────
        self._cmd_pub = rospy.Publisher(
            "/local_waypoint_cmd", PointStamped, queue_size=1
        )

        # ── 订阅器 ────────────────────────────────────────
        rospy.Subscriber("/fix", NavSatFix, self._gps_callback)
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/planned_path", Path, self._path_callback)
        rospy.Subscriber(
            "/local_waypoint_reached", Bool, self._reached_callback
        )
        rospy.Subscriber(
            "/yaw_alignment", Float64, self._yaw_align_callback
        )

        rospy.loginfo("=" * 55)
        rospy.loginfo(" GPS Waypoint Navigator (纯导航层)")
        rospy.loginfo("  订阅: /fix, /odom, /planned_path,")
        rospy.loginfo("         /local_waypoint_reached, /yaw_alignment")
        rospy.loginfo("  发布: /local_waypoint_cmd")
        rospy.loginfo("  GPS 到达阈值: %.1f m", GPS_REACH_THRESHOLD)
        rospy.loginfo("  路径点超时:   %.1f s", WAYPOINT_TIMEOUT)
        rospy.loginfo("  指令频率:     %d Hz", CTRL_CMD_RATE)
        rospy.loginfo("=" * 55)

    # ══════════════════════════════════════════════════════
    #  回调函数
    # ══════════════════════════════════════════════════════

    def _yaw_align_callback(self, msg: Float64):
        """接收并缓存对齐角度"""
        self._yaw_correction = msg.data
        if not self._yaw_align_received:
            self._yaw_align_received = True
            rospy.loginfo("[导航] ✅ 收到对齐角度: %.2f°",
                          math.degrees(self._yaw_correction))

    def _gps_callback(self, msg: NavSatFix):
        """缓存最新有效的 GPS 位置 (WGS84)"""
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        if msg.status.status < 0:
            return
        self._current_gps = (msg.latitude, msg.longitude)

    def _odom_callback(self, msg: Odometry):
        """缓存最新里程计"""
        self._current_odom = msg

    def _path_callback(self, msg: Path):
        """接收规划路径（GCJ-02 坐标点列表）"""
        if self._state not in (self.STATE_IDLE, self.STATE_COMPLETE):
            rospy.logwarn("[导航] 正在导航中，忽略新路径 (state=%s)", self._state)
            return

        if not msg.poses:
            rospy.logwarn("[导航] 收到空路径，忽略")
            return

        self._waypoints = []
        for pose in msg.poses:
            lng = pose.pose.position.x   # 经度 GCJ-02
            lat = pose.pose.position.y   # 纬度 GCJ-02
            self._waypoints.append((lat, lng))

        self._current_wp_idx = 0
        self._state = self.STATE_PATH_READY

        rospy.loginfo("=" * 55)
        rospy.loginfo("  ✅ 收到路径规划: %d 个路径点", len(self._waypoints))
        rospy.loginfo("  起点: (%.6f, %.6f)", self._waypoints[0][0], self._waypoints[0][1])
        rospy.loginfo("  终点: (%.6f, %.6f)", self._waypoints[-1][0], self._waypoints[-1][1])
        rospy.loginfo("=" * 55)

    def _reached_callback(self, msg: Bool):
        """来自 target_point 的到达信号"""
        if msg.data:
            self._reached_signal = True
            self._reached_signal_time = rospy.Time.now()

    # ══════════════════════════════════════════════════════
    #  工具函数
    # ══════════════════════════════════════════════════════

    def _send_cmd(self, x: float, y: float):
        """发送相对位姿指令给运动层"""
        cmd = PointStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "base_link"
        cmd.point.x = x
        cmd.point.y = y
        self._cmd_pub.publish(cmd)

    def _stop_robot(self):
        """发送停止指令"""
        self._send_cmd(0.0, 0.0)

    def _compute_target_relative(self, wp_idx: int):
        """
        计算目标点在机器人坐标系下的相对位姿。

        利用：
          - 当前 GPS (WGS84)  → 机器人位置
          - 目标点 (GCJ-02)   → 目标位置
          - odom yaw + 对齐角度 → 机器人真实朝向

        返回: (x_forward, y_left, distance, arrived)
        """
        robot_lat, robot_lon = self._current_gps  # WGS84
        target_lat_gcj, target_lon_gcj = self._waypoints[wp_idx]

        try:
            raw_yaw = get_yaw_from_odom(self._current_odom)
            # 应用对齐修正 → 得到真实世界朝向
            corrected_yaw = raw_yaw + self._yaw_correction
            corrected_yaw = math.atan2(
                math.sin(corrected_yaw), math.cos(corrected_yaw)
            )
        except Exception as e:
            rospy.logwarn_throttle(3.0, "[导航] 提取 yaw 失败: %s", e)
            return 0.0, 0.0, 999.0, False

        x_forward, y_left, distance, reached = get_target_in_robot_frame(
            robot_lat, robot_lon,
            corrected_yaw,
            target_lat_gcj, target_lon_gcj,
            reach_threshold=GPS_REACH_THRESHOLD
        )
        return x_forward, y_left, distance, reached

    # ══════════════════════════════════════════════════════
    #  导航控制
    # ══════════════════════════════════════════════════════

    def _start_waypoint(self, idx: int):
        """切换到指定路径点"""
        self._current_wp_idx = idx
        self._wp_start_time = rospy.Time.now()
        self._reached_signal = False
        self._reached_signal_time = rospy.Time(0)
        self._gps_arrived_flag = False

        lat_gcj, lon_gcj = self._waypoints[idx]
        rospy.loginfo("─" * 40)
        rospy.loginfo(
            "  🚀 开始路径点 [%d/%d]: (%.6f, %.6f)",
            idx + 1, len(self._waypoints), lat_gcj, lon_gcj
        )

    def _advance_or_complete(self):
        """推进到下一路径点，或标记完成"""
        next_idx = self._current_wp_idx + 1
        if next_idx >= len(self._waypoints):
            self._state = self.STATE_COMPLETE
            rospy.loginfo("=" * 40)
            rospy.loginfo("  🏁 导航完成！所有路径点已到达！")
            rospy.loginfo("=" * 40)
        else:
            self._state = self.STATE_FOLLOWING
            self._start_waypoint(next_idx)

    # ══════════════════════════════════════════════════════
    #  主循环
    # ══════════════════════════════════════════════════════

    def run(self):
        """主导航循环"""
        rate = rospy.Rate(CTRL_CMD_RATE)

        # ── Phase 1: 等待对齐角度 ───────────────────────────
        rospy.loginfo("=" * 50)
        rospy.loginfo("  Phase 1: 等待对齐角度 (/yaw_alignment) ...")
        rospy.loginfo("=" * 50)

        align_wait_start = rospy.Time.now()
        while not rospy.is_shutdown() and not self._yaw_align_received:
            elapsed = (rospy.Time.now() - align_wait_start).to_sec()
            if elapsed > ALIGN_WAIT_TIMEOUT:
                rospy.logwarn(
                    "[导航] 对齐角度等待超时 (%.1fs)，使用原始 odom yaw", elapsed
                )
                break
            rate.sleep()

        if self._yaw_align_received:
            rospy.loginfo("[导航] ✅ 对齐角度已接收: %.2f°",
                          math.degrees(self._yaw_correction))
        else:
            rospy.logwarn("[导航] ⚠️ 未收到对齐角度，将使用原始 odom yaw (修正=0)")

        # ── Phase 2: 等待路径 ───────────────────────────────
        rospy.loginfo("=" * 50)
        rospy.loginfo("  Phase 2: 等待路径规划 (/planned_path) ...")
        rospy.loginfo("=" * 50)

        # ── Phase 3: 主导航循环 ────────────────────────────
        while not rospy.is_shutdown():

            # ── IDLE ──────────────────────────────────────
            if self._state == self.STATE_IDLE:
                rate.sleep()
                continue

            # ── PATH_READY → 检查传感器 → FOLLOWING ──────
            if self._state == self.STATE_PATH_READY:
                if self._current_gps is None or self._current_odom is None:
                    rospy.logwarn_throttle(
                        3.0, "[导航] ⏳ 等待传感器就绪..."
                    )
                    rate.sleep()
                    continue
                rospy.loginfo("[导航] ✅ 传感器就绪，开始导航")
                self._state = self.STATE_FOLLOWING
                self._start_waypoint(0)
                continue

            # ── COMPLETE ──────────────────────────────────
            if self._state == self.STATE_COMPLETE:
                self._stop_robot()
                rate.sleep()
                continue

            # ── TRANSITION → advance ──────────────────────
            if self._state == self.STATE_TRANSITION:
                self._advance_or_complete()
                continue

            # ── FOLLOWING ─────────────────────────────────
            if self._state != self.STATE_FOLLOWING:
                rate.sleep()
                continue

            # ── 安全检查 ──────────────────────────────────
            if self._current_gps is None or self._current_odom is None:
                rospy.logwarn_throttle(
                    2.0, "[导航] ⚠️ 传感器丢失，停止运动"
                )
                self._stop_robot()
                rate.sleep()
                continue

            # ── 计算当前相对位姿 & GPS 到达检查 ──────────
            x_f, y_l, dist, gps_arrived = self._compute_target_relative(
                self._current_wp_idx
            )

            wp_label = f"{self._current_wp_idx + 1}/{len(self._waypoints)}"
            now = rospy.Time.now()
            elapsed = (now - self._wp_start_time).to_sec()

            # ── 条件1: 超时 ──────────────────────────────
            if elapsed > WAYPOINT_TIMEOUT:
                rospy.logwarn(
                    "  ⏰ [%s] 超时 (%.1fs), 强制跳过",
                    wp_label, elapsed
                )
                self._stop_robot()
                self._state = self.STATE_TRANSITION
                continue

            # ── 条件2: GPS 到达（带迟滞，防噪声震动） ────
            # 一旦抵达就锁定，不会因GPS噪声起伏而反复切换
            if not self._gps_arrived_flag:
                self._gps_arrived_flag = (gps_arrived or dist < GPS_REACH_THRESHOLD)
            if self._gps_arrived_flag:
                rospy.loginfo(
                    "  ✅ [%s] GPS 到达! 距离=%.2fm, 耗时=%.1fs",
                    wp_label, dist, elapsed
                )
                self._stop_robot()
                self._state = self.STATE_TRANSITION
                continue

            # ── 条件3: 运动层到达信号 ────────────────────
            # 只有信号是在当前航点启动之后到达的才视为有效，防止上一航点的信号串扰
            signal_age = (now - self._reached_signal_time).to_sec()
            if self._reached_signal and elapsed > 1.5 and signal_age < 1.0:
                rospy.loginfo(
                    "  ✅ [%s] 控制层到达信号! 距离=%.2fm, 耗时=%.1fs",
                    wp_label, dist, elapsed
                )
                self._stop_robot()
                self._state = self.STATE_TRANSITION
                continue

            # ── 正常发送指令给运动层 ────────────────────
            self._send_cmd(x_f, y_l)

            # ── 降频日志 ─────────────────────────────────
            if int(now.to_sec()) % 3 == 0 \
               and (now - self._last_cmd_time).to_sec() > 1.0:
                tgt_lat, tgt_lon = self._waypoints[self._current_wp_idx]
                rospy.loginfo(
                    "  🚀 [%s] rel=(%.2f, %.2f)m | dist=%.2fm | "
                    "yaw_corr=%.1f° | t=%.1fs",
                    wp_label, x_f, y_l, dist,
                    math.degrees(self._yaw_correction),
                    elapsed
                )
                self._last_cmd_time = now

            rate.sleep()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        navigator = GPSWaypointNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[gps_navigator] 节点已退出")
    except Exception as e:
        rospy.logerr("[gps_navigator] 未预期错误: %s", e)
        import traceback
        traceback.print_exc()
