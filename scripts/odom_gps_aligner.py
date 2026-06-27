#!/usr/bin/env python3
"""
odom_gps_aligner.py — Odom 与 GPS(UTM) 角度对齐节点 (testgps)
────────────────────────────────────────────────────────────
功能：
  1. 等传感器就绪后，发布 /alignment_mode=True 让对齐运动节点匀速直走
  2. 记录起始 GPS(UTM) 和 odom 位姿
  3. 通过 odom 监测前进距离，达到 5m 后通知停止
  4. 记录结束 GPS(UTM) 和 odom 位姿
  5. 计算对齐角度 (yaw_correction = gps_angle - odom_angle)
  6. 持续发布 /yaw_alignment 供导航层使用

通信：
  Pub: /alignment_mode     (std_msgs/Bool)    对齐模式开关 → alignment_motion
  Pub: /yaw_alignment      (std_msgs/Float64) 对齐角度（弧度）→ gps_waypoint_navigator
  Pub: /local_waypoint_cmd (geometry_msgs/PointStamped) 停止指令
  Sub: /fix                (sensor_msgs/NavSatFix)  GPS WGS84
  Sub: /odom               (nav_msgs/Odometry)      轮式里程计
"""

import rospy
import math
import sys
import os

from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import PointStamped

# ── 动态导入坐标转换模块 ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from coord_transform_robot import get_yaw_from_odom, TRANSFORMER
except ImportError as e:
    print(f"[odom_gps_aligner] 无法导入 coord_transform_robot: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════
ALIGNMENT_MOVE_DISTANCE = 5.0    # 对齐前进距离（米）
SENSOR_WAIT_TIMEOUT     = 10.0   # 等待传感器超时 (s)
ALIGNMENT_TIMEOUT       = 30.0   # 对齐运动超时 (s)
YAW_PUB_RATE            = 5      # 对齐角度发布频率 (Hz)

LOOP_RATE               = 10     # 主循环频率 (Hz)


class OdomGPSAligner:
    """
    Odom ↔ GPS(UTM) 角度对齐器

    状态机:
      IDLE → COLLECT_INIT → MOVING → COLLECT_END → COMPLETE
    """

    STATE_IDLE         = "IDLE"
    STATE_COLLECT_INIT = "COLLECT_INIT"
    STATE_MOVING       = "MOVING"
    STATE_COLLECT_END  = "COLLECT_END"
    STATE_COMPLETE     = "COMPLETE"
    STATE_FAILED       = "FAILED"

    def __init__(self):
        rospy.init_node("odom_gps_aligner", anonymous=False)

        # ── 状态 ────────────────────────────────────────────
        self._state = self.STATE_IDLE
        self._current_gps = None          # (lat_wgs84, lon_wgs84)
        self._current_odom = None

        # ── 对齐记录 ────────────────────────────────────────
        self._init_gps_utm  = None        # (x, y) UTM
        self._init_odom_pose = None       # (x, y, yaw)
        self._end_gps_utm   = None
        self._end_odom_pose = None
        self._yaw_correction = 0.0        # 最终对齐角度（弧度）

        # ── 发布器 ──────────────────────────────────────────
        # latch=True 确保晚订阅的节点也能收到最近一次状态
        self._align_mode_pub = rospy.Publisher(
            "/alignment_mode", Bool, queue_size=1, latch=True
        )
        self._yaw_align_pub = rospy.Publisher(
            "/yaw_alignment", Float64, queue_size=1
        )
        self._stop_pub = rospy.Publisher(
            "/local_waypoint_cmd", PointStamped, queue_size=1
        )

        # ── 订阅器 ──────────────────────────────────────────
        rospy.Subscriber("/fix", NavSatFix, self._gps_callback)
        rospy.Subscriber("/odom", Odometry, self._odom_callback)

        rospy.loginfo("=" * 50)
        rospy.loginfo(" Odom-GPS Aligner")
        rospy.loginfo(" 订阅: /fix, /odom")
        rospy.loginfo(" 发布: /alignment_mode, /yaw_alignment")
        rospy.loginfo(" 前进 %.1f m 以对齐 Odom ↔ GPS(UTM)", ALIGNMENT_MOVE_DISTANCE)
        rospy.loginfo("=" * 50)

    # ══════════════════════════════════════════════════════
    #  回调
    # ══════════════════════════════════════════════════════
    def _gps_callback(self, msg: NavSatFix):
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        if msg.status.status < 0:
            return
        self._current_gps = (msg.latitude, msg.longitude)

    def _odom_callback(self, msg: Odometry):
        self._current_odom = msg

    # ══════════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════════
    def _get_odom_pose(self):
        """返回 (x, y, yaw) 或 None"""
        if not self._current_odom:
            return None
        x = self._current_odom.pose.pose.position.x
        y = self._current_odom.pose.pose.position.y
        yaw = get_yaw_from_odom(self._current_odom)
        return (x, y, yaw)

    def _gps_to_utm(self, lat, lon):
        x, y = TRANSFORMER.transform(lon, lat)
        return (x, y)

    def _send_stop(self):
        """发送停止指令"""
        cmd = PointStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "base_link"
        cmd.point.x = 0.0
        cmd.point.y = 0.0
        self._stop_pub.publish(cmd)

    def _wait_sensors(self, rate: rospy.Rate) -> bool:
        """等待 GPS 和 Odom 就绪"""
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self._current_gps and self._current_odom:
                return True
            elapsed = (rospy.Time.now() - start).to_sec()
            if elapsed > SENSOR_WAIT_TIMEOUT:
                rospy.logerr("[对齐] 传感器等待超时 (%.1fs)", elapsed)
                return False
            if not self._current_gps:
                rospy.logwarn_throttle(2.0, "[对齐] ⏳ 等待 GPS...")
            if not self._current_odom:
                rospy.logwarn_throttle(2.0, "[对齐] ⏳ 等待 Odom...")
            rate.sleep()
        return False

    def _sample_gps_utm(self, label: str, timeout=5.0, samples=10) -> tuple:
        """采集多次 GPS 取平均，返回 (x_utm, y_utm)，大幅降低单点噪声"""
        collected = []
        start = rospy.Time.now()

        # 先等第一个有效 GPS
        while not rospy.is_shutdown() and not self._current_gps:
            if (rospy.Time.now() - start).to_sec() > timeout:
                raise TimeoutError(f"{label} GPS 采集超时（无信号）")
            rospy.sleep(0.05)

        # 连续采集 samples 个点
        while not rospy.is_shutdown() and len(collected) < samples:
            if self._current_gps:
                lat, lon = self._current_gps
                x, y = self._gps_to_utm(lat, lon)
                collected.append((x, y))
            if (rospy.Time.now() - start).to_sec() > timeout:
                break
            rospy.sleep(0.1)  # 10Hz

        if len(collected) < 3:
            raise TimeoutError(f"{label} GPS 采集超时（仅 {len(collected)} 个点）")

        avg_x = sum(p[0] for p in collected) / len(collected)
        avg_y = sum(p[1] for p in collected) / len(collected)
        return (avg_x, avg_y)

    # ══════════════════════════════════════════════════════
    #  对齐流程
    # ══════════════════════════════════════════════════════
    def _do_alignment(self, rate: rospy.Rate) -> bool:
        """执行对齐流程，成功返回 True"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("  开始 Odom ↔ GPS(UTM) 对齐")
        rospy.loginfo("  直线前进 %.1f 米", ALIGNMENT_MOVE_DISTANCE)
        rospy.loginfo("=" * 50)

        # ── 1. 采集初始点 ────────────────────────────────────
        self._state = self.STATE_COLLECT_INIT
        try:
            self._init_gps_utm = self._sample_gps_utm("初始", timeout=5.0)
        except TimeoutError as e:
            rospy.logerr("[对齐] %s", e)
            self._state = self.STATE_FAILED
            return False

        init_pose = self._get_odom_pose()
        if init_pose is None:
            rospy.logerr("[对齐] 无法获取初始 odom")
            self._state = self.STATE_FAILED
            return False
        self._init_odom_pose = init_pose

        rospy.loginfo("  初始 GPS(UTM): (%.1f, %.1f)",
                      self._init_gps_utm[0], self._init_gps_utm[1])
        rospy.loginfo("  初始 odom:      (%.2f, %.2f, yaw=%.1f°)",
                      self._init_odom_pose[0], self._init_odom_pose[1],
                      math.degrees(self._init_odom_pose[2]))

        # ── 2. 开始前进 ─────────────────────────────────────
        self._state = self.STATE_MOVING
        self._align_mode_pub.publish(Bool(True))
        rospy.sleep(0.3)  # 确保运动节点收到

        init_x, init_y = self._init_odom_pose[0], self._init_odom_pose[1]
        move_start = rospy.Time.now()
        traveled = 0.0
        last_log_time = rospy.Time.now()

        while not rospy.is_shutdown():
            pose = self._get_odom_pose()
            if pose:
                traveled = math.hypot(pose[0] - init_x, pose[1] - init_y)

            if traveled >= ALIGNMENT_MOVE_DISTANCE:
                rospy.loginfo("  已前进 %.2f / %.1f m", traveled, ALIGNMENT_MOVE_DISTANCE)
                break

            elapsed = (rospy.Time.now() - move_start).to_sec()
            if elapsed > ALIGNMENT_TIMEOUT:
                rospy.logerr("[对齐] 移动超时 (%.1fs, 仅走了 %.2f m)", elapsed, traveled)
                self._state = self.STATE_FAILED
                break

            # 每 2s 打印进度
            now = rospy.Time.now()
            if (now - last_log_time).to_sec() > 2.0:
                rospy.loginfo("  对齐中 … 已前进 %.2f / %.1f m (%.1fs)",
                              traveled, ALIGNMENT_MOVE_DISTANCE, elapsed)
                last_log_time = now

            rate.sleep()

        # ── 3. 停止 ─────────────────────────────────────────
        self._align_mode_pub.publish(Bool(False))
        rospy.sleep(0.1)
        self._send_stop()
        rospy.sleep(0.5)

        if self._state == self.STATE_FAILED:
            return False

        # ── 4. 采集结束点 ───────────────────────────────────
        self._state = self.STATE_COLLECT_END
        try:
            self._end_gps_utm = self._sample_gps_utm("结束", timeout=5.0)
        except TimeoutError as e:
            rospy.logerr("[对齐] %s", e)
            self._state = self.STATE_FAILED
            return False

        end_pose = self._get_odom_pose()
        if end_pose is None:
            rospy.logerr("[对齐] 无法获取结束 odom")
            self._state = self.STATE_FAILED
            return False
        self._end_odom_pose = end_pose

        rospy.loginfo("  结束 GPS(UTM): (%.1f, %.1f)",
                      self._end_gps_utm[0], self._end_gps_utm[1])
        rospy.loginfo("  结束 odom:      (%.2f, %.2f, yaw=%.1f°)",
                      self._end_odom_pose[0], self._end_odom_pose[1],
                      math.degrees(self._end_odom_pose[2]))

        # ── 5. 计算对齐角度 ─────────────────────────────────
        gps_dx = self._end_gps_utm[0] - self._init_gps_utm[0]
        gps_dy = self._end_gps_utm[1] - self._init_gps_utm[1]
        gps_angle = math.atan2(gps_dy, gps_dx)

        odom_dx = self._end_odom_pose[0] - self._init_odom_pose[0]
        odom_dy = self._end_odom_pose[1] - self._init_odom_pose[1]
        odom_angle = math.atan2(odom_dy, odom_dx)

        self._yaw_correction = gps_angle - odom_angle
        # 归一化到 [-π, π]
        self._yaw_correction = math.atan2(
            math.sin(self._yaw_correction),
            math.cos(self._yaw_correction)
        )

        rospy.loginfo("=" * 50)
        rospy.loginfo("  对齐完成!")
        rospy.loginfo("  GPS 路径角度:  %.2f°", math.degrees(gps_angle))
        rospy.loginfo("  Odom 路径角度: %.2f°", math.degrees(odom_angle))
        rospy.loginfo("  yaw 修正量:    %.2f°", math.degrees(self._yaw_correction))
        rospy.loginfo("=" * 50)

        self._state = self.STATE_COMPLETE
        return True

    # ══════════════════════════════════════════════════════
    #  主循环
    # ══════════════════════════════════════════════════════
    def run(self):
        rate = rospy.Rate(LOOP_RATE)

        # ── Phase 1: 等传感器 + 执行对齐 ────────────────────
        rospy.loginfo("[对齐] 等待传感器就绪...")
        if not self._wait_sensors(rate):
            rospy.logerr("[对齐] 传感器未就绪，将对齐角度设为 0，继续运行")
            self._yaw_correction = 0.0
            self._state = self.STATE_COMPLETE
        else:
            rospy.loginfo("[对齐] 传感器就绪，执行对齐")
            if not self._do_alignment(rate):
                rospy.logwarn("[对齐] 对齐流程未完成，将对齐角度设为 0")
                self._yaw_correction = 0.0
                self._state = self.STATE_COMPLETE

        # ── Phase 2: 持续发布对齐角度 ───────────────────────
        rospy.loginfo("[对齐] 持续发布 yaw 修正: %.2f°",
                      math.degrees(self._yaw_correction))

        yaw_rate = rospy.Rate(YAW_PUB_RATE)
        while not rospy.is_shutdown():
            self._yaw_align_pub.publish(Float64(self._yaw_correction))
            yaw_rate.sleep()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        node = OdomGPSAligner()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[对齐] 节点退出")
    except Exception as e:
        rospy.logerr("[对齐] 错误: %s", e)
        import traceback
        traceback.print_exc()
