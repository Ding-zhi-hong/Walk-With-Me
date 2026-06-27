#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于时间窗口的动态回溯采样缓存器
输出预处理好的resize图像 + 基于compute_relative_pose计算的相对位姿
"""
import rospy
import cv2
from collections import deque
import threading
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import os

# ================= 动态路径导入坐标变换工具 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

try:
    from coord_transform_robot import compute_relative_pose
except ImportError as e:
    rospy.logerr(f"❌ 导入 coord_transform_robot 失败，请确保文件在同一目录下: {e}")
    raise e


class TimeWindowCache:
    def __init__(self, window_seconds: float = 10.0, out_img_w=630, out_img_h=350):
        self.window_duration = rospy.Duration(window_seconds)
        self.out_w = out_img_w
        self.out_h = out_img_h
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.odom_buffer = deque()
        self.image_buffer = deque()

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.image_sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback)

        rospy.loginfo(f"✅ {window_seconds}秒滑窗缓存初始化完成，输出图像尺寸 {out_img_w}×{out_img_h}")

    def odom_callback(self, msg: Odometry):
        t = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        with self.lock:
            self.odom_buffer.append((t, msg))
            while self.odom_buffer and (t - self.odom_buffer[0][0]) > self.window_duration:
                self.odom_buffer.popleft()

    def image_callback(self, msg: Image):
        t = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换失败: {e}")
            return
        with self.lock:
            self.image_buffer.append((t, cv_img))
            while self.image_buffer and (t - self.image_buffer[0][0]) > self.window_duration:
                self.image_buffer.popleft()

    def get_all_cached_data(self):
        with self.lock:
            return list(self.odom_buffer), list(self.image_buffer)

    def _find_nearest_by_time(self, buf: list, target_t: rospy.Time, max_diff_sec=0.2):
        if not buf:
            return None
        best_item = None
        min_dt = float("inf")
        for t, data in buf:
            dt = abs((t - target_t).to_sec())
            if dt < min_dt:
                min_dt = dt
                best_item = data
        if min_dt > max_diff_sec:
            return None
        return best_item

    def sample_5_1hz_pairs(self):
        """原始采样：返回5组原始odom、原图"""
        with self.lock:
            odom_buf = list(self.odom_buffer)
            img_buf = list(self.image_buffer)
            if len(odom_buf) == 0 or len(img_buf) == 0:
                rospy.logwarn_throttle(2, "缓存无数据，无法采样")
                return None
            latest_time = odom_buf[-1][0]

        sample_odoms = []
        sample_imgs = []
        for offset_sec in [0, 1, 2, 3, 4]:
            target_t = latest_time - rospy.Duration(offset_sec)
            odom_data = self._find_nearest_by_time(odom_buf, target_t)
            img_data = self._find_nearest_by_time(img_buf, target_t)
            if odom_data is None or img_data is None:
                rospy.logwarn_throttle(2, f"偏移{offset_sec}s匹配失败，本次采样作废")
                return None
            sample_odoms.append(odom_data)
            sample_imgs.append(img_data)

        sample_odoms.reverse()
        sample_imgs.reverse()
        return sample_odoms, sample_imgs

    def get_citywalker_input(self):
        """
        新增处理函数：
        1. 调用sample_5_1hz_pairs拿到原始5组数据
        2. 图像resize到630×350并深拷贝
        3. 使用compute_relative_pose计算每组相对xy
        return: (past_poses: List[(float, float)], processed_imgs: List[np.ndarray])
        顺序 [T-4, T-3, T-2, T-1, T0]
        采样失败返回 None
        """
        raw_res = self.sample_5_1hz_pairs()
        if raw_res is None:
            return None
        odom_list, raw_img_list = raw_res

        # 1. 图像统一resize
        processed_imgs = []
        for img in raw_img_list:
            resize_img = cv2.resize(img, (self.out_w, self.out_h), cv2.INTER_LINEAR)
            processed_imgs.append(resize_img)

        # 2. 提取当前帧(T0)位姿作为基准
        cur_odom = odom_list[-1]
        cur_p = cur_odom.pose.pose.position
        cur_o = cur_odom.pose.pose.orientation
        cur_pose_arr = [cur_p.x, cur_p.y, cur_p.z, cur_o.x, cur_o.y, cur_o.z, cur_o.w]

        past_poses = []
        for odom_msg in odom_list:
            p = odom_msg.pose.pose.position
            o = odom_msg.pose.pose.orientation
            tgt_pose_arr = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
            # 调用外部导入的compute_relative_pose
            rel_pos, _ = compute_relative_pose(cur_pose_arr, tgt_pose_arr)
            past_poses.append((float(rel_pos[0]), float(rel_pos[1])))

        return past_poses, processed_imgs