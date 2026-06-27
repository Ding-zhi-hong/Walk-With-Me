#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换工具（模块化接口）
功能：
    输入机器人GPS(WGS84)、IMU yaw、目标点(GCJ-02)
    输出目标点在机器人坐标系下的 (x, y)
    支持从 odometry 消息提取 yaw 角度

坐标系定义：
    - x > 0 : 目标在机器人前方
    - y > 0 : 目标在机器人左侧
    - yaw = 0 : 车头朝正东（与ROS标准保持一致）
"""
from scipy.spatial.transform import Rotation as R
import math
import pyproj
import sys
import os
from typing import Tuple, Optional
from typing import Tuple, List, Optional
import numpy as np
# ================= 动态路径导入 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from coordTransform_utils import gcj02_to_wgs84
except ImportError as e:
    raise ImportError(
        f"无法导入 coordTransform_utils: {e}\n"
        f"请确保该文件位于: {SCRIPT_DIR}"
    )

# ================= UTM 转换器（合肥 Zone 50） =================
TRANSFORMER = pyproj.Transformer.from_crs(
    "EPSG:4326",  # WGS84
    "+proj=utm +zone=50 +ellps=WGS84 +units=m +no_defs",
    always_xy=True
)



def get_yaw_from_odom(odom_msg) -> float:
    """
    从 odometry 消息中提取 yaw 角度（弧度制）
    
    参数:
        odom_msg: nav_msgs/Odometry 消息
        
    返回:
        float: yaw 角度（弧度制），0=正东方向
    """
    # 从 odometry 消息中提取四元数
    orientation = odom_msg.pose.pose.orientation
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    
    # 四元数转欧拉角（yaw）
    # 使用标准公式：yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    yaw = math.atan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    
    return yaw


def get_yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """
    从四元数直接提取 yaw 角度（弧度制）
    
    参数:
        x, y, z, w: 四元数分量
        
    返回:
        float: yaw 角度（弧度制），0=正东方向
    """
    # 四元数转欧拉角（yaw）
    yaw = math.atan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    
    return yaw


def get_target_in_robot_frame(
    robot_lat_wgs: float,
    robot_lon_wgs: float,
    robot_yaw_rad: float,
    target_lat_gcj: float,
    target_lon_gcj: float,
    reach_threshold: float = 1.0
) -> Tuple[float, float, float, bool]:
    """
    计算目标点相对于机器人坐标系的位置

    参数:
        robot_lat_wgs (float): 机器人纬度 (WGS84)
        robot_lon_wgs (float): 机器人经度 (WGS84)
        robot_yaw_rad (float): 机器人朝向 (弧度制, 0=正东)
        target_lat_gcj (float): 目标纬度 (GCJ-02 / 高德)
        target_lon_gcj (float): 目标经度 (GCJ-02 / 高德)
        reach_threshold (float): 到达判定阈值（米）

    返回:
        tuple:
        (
            x_forward (float),  # 前向距离 (m), >0 为前
            y_left (float),    # 侧向距离 (m), >0 为左
            distance (float),  # 直线距离 (m)
            reached (bool)     # 是否到达目标点
        )
    """

    # ---------- 1. 目标 GCJ-02 → WGS84 ----------
    tgt_lon_wgs, tgt_lat_wgs = gcj02_to_wgs84(target_lon_gcj, target_lat_gcj)

    # ---------- 2. WGS84 → UTM ----------
    robot_x, robot_y = TRANSFORMER.transform(robot_lon_wgs, robot_lat_wgs)
    tgt_x, tgt_y = TRANSFORMER.transform(tgt_lon_wgs, tgt_lat_wgs)

    # ---------- 3. 世界坐标系相对位移 ----------
    dx = tgt_x - robot_x  # 东向
    dy = tgt_y - robot_y  # 北向

    # ---------- 4. 旋转至机器人坐标系 ----------
    cos_y = math.cos(robot_yaw_rad)
    sin_y = math.sin(robot_yaw_rad)

    x_forward = dx * cos_y + dy * sin_y
    y_left = -dx * sin_y + dy * cos_y

    # ---------- 5. 到达判定 ----------
    distance = math.hypot(x_forward, y_left)
    reached = distance < reach_threshold

    return x_forward, y_left, distance, reached


def get_target_in_robot_frame_from_odom(
    robot_lat_wgs: float,
    robot_lon_wgs: float,
    odom_msg,
    target_lat_gcj: float,
    target_lon_gcj: float,
    reach_threshold: float = 1.0,
    yaw_correction: float = 0.0
) -> Tuple[float, float, float, bool]:
    """
    直接从 odometry 消息计算目标点相对于机器人坐标系的位置（便捷函数）

    参数:
        robot_lat_wgs (float): 机器人纬度 (WGS84)
        robot_lon_wgs (float): 机器人经度 (WGS84)
        odom_msg: nav_msgs/Odometry 消息（包含姿态信息）
        target_lat_gcj (float): 目标纬度 (GCJ-02 / 高德)
        target_lon_gcj (float): 目标经度 (GCJ-02 / 高德)
        reach_threshold (float): 到达判定阈值（米）
        yaw_correction (float): Odom与GPS(UTM)的对齐角度修正（弧度），
                                由 odom_gps_aligner 计算并通过 /yaw_alignment 发布

    返回:
        tuple:
        (
            x_forward (float),  # 前向距离 (m), >0 为前
            y_left (float),    # 侧向距离 (m), >0 为左
            distance (float),  # 直线距离 (m)
            reached (bool)     # 是否到达目标点
        )
    """
    # 从 odometry 提取 yaw
    robot_yaw_rad = get_yaw_from_odom(odom_msg)

    # 应用对齐修正（Odom → GPS(UTM) 方向对齐）
    if abs(yaw_correction) > 1e-10:
        robot_yaw_rad = robot_yaw_rad + yaw_correction
        robot_yaw_rad = math.atan2(
            math.sin(robot_yaw_rad),
            math.cos(robot_yaw_rad)
        )

    # 调用主函数
    return get_target_in_robot_frame(
        robot_lat_wgs,
        robot_lon_wgs,
        robot_yaw_rad,
        target_lat_gcj,
        target_lon_gcj,
        reach_threshold
    )

def quaternion_to_matrix(q):
    """
    将四元数转换为旋转矩阵
    四元数格式: [qx, qy, qz, qw]
    """
    rot = R.from_quat([q[0], q[1], q[2], q[3]])
    return rot.as_matrix()

def pose_to_matrix(pose):
    """
    将位姿转换为4x4变换矩阵
    pose格式: [x, y, z, qx, qy, qz, qw]
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = quaternion_to_matrix(pose[3:])
    return T

def compute_relative_pose(current_pose, target_pose):
    """
    计算目标点相对于当前点的相对位姿
    """
    T_cur = pose_to_matrix(current_pose)
    T_tgt = pose_to_matrix(target_pose)
    
    # 计算相对变换: T_rel = T_cur^{-1} * T_tgt
    T_rel = np.linalg.inv(T_cur) @ T_tgt
    
    # 提取相对位置和旋转
    rel_pos = T_rel[:3, 3]
    rel_rot = T_rel[:3, :3]
    
    return rel_pos, rel_rot






