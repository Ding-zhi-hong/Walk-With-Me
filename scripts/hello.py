#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用TimeWindowCache类并实时可视化输出的示例代码
"""
import rospy
import cv2
import numpy as np
import threading
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


from data_collector import TimeWindowCache  




class CacheVisualizer:
    def __init__(self, window_seconds=10.0, img_w=630, img_h=350):
        # 初始化缓存器
        self.cache = TimeWindowCache(window_seconds=window_seconds, 
                                     out_img_w=img_w, 
                                     out_img_h=img_h)
        self.visualize_window_name = "CityWalker Input (T-4 ~ T0)"
        self.is_ready = False
        self.lock = threading.Lock()
        
        # 初始化可视化窗口
        cv2.namedWindow(self.visualize_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.visualize_window_name, 1280, 720)

    def wait_for_cache_ready(self):
        """等待缓存中充满足够的数据（至少5秒的odom和图像数据）"""
        rospy.loginfo("⏳ 等待缓存填充数据...")
        rate = rospy.Rate(1)  # 1Hz检查频率
        
        while not rospy.is_shutdown():
            odom_buf, img_buf = self.cache.get_all_cached_data()
            
            # 检查缓存数据量是否足够（至少覆盖5秒的时间窗口）
            if len(odom_buf) > 0 and len(img_buf) > 0:
                # 检查最早的缓存数据是否至少覆盖5秒
                odom_time_span = (odom_buf[-1][0] - odom_buf[0][0]).to_sec()
                img_time_span = (img_buf[-1][0] - img_buf[0][0]).to_sec()
                
                if odom_time_span >= 5.0 and img_time_span >= 5.0:
                    # 预采样一次验证是否能成功获取5组数据
                    sample_data = self.cache.get_citywalker_input()
                    if sample_data is not None:
                        rospy.loginfo("✅ 缓存数据充足，可以开始可视化！")
                        with self.lock:
                            self.is_ready = True
                        break
            
            rospy.loginfo_throttle(5, f"缓存进度：里程计{len(odom_buf)}条 | 图像{len(img_buf)}条 | 需至少覆盖5秒数据")
            rate.sleep()

    def visualize_data(self):
        """实时可视化get_citywalker_input返回的5张图片和位姿信息"""
        rate = rospy.Rate(10)  # 10Hz可视化频率
        
        while not rospy.is_shutdown() and self.is_ready:
            # 获取处理后的城市漫步输入数据
            data = self.cache.get_citywalker_input()
            
            if data is None:
                rospy.logwarn_throttle(1, "⚠️ 采样失败，跳过本次可视化")
                rate.sleep()
                continue
            
            past_poses, processed_imgs = data
            
            # 拼接5张图片为一行，方便可视化
            # 每张图添加文字标注（时间戳和相对位姿）
            annotated_imgs = []
            time_labels = ["T-4", "T-3", "T-2", "T-1", "T0"]
            
            for idx, (img, pose, label) in enumerate(zip(processed_imgs, past_poses, time_labels)):
                # 深拷贝避免修改原数据
                img_copy = img.copy()
                h, w = img_copy.shape[:2]
                
                # 添加文字标注
                text = f"{label}: (x={pose[0]:.3f}, y={pose[1]:.3f})"
                cv2.putText(img_copy, text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 调整单张图片高度，统一拼接
                annotated_imgs.append(img_copy)
            
            # 横向拼接5张图片
            combined_img = np.hstack(annotated_imgs)
            
            # 显示拼接后的图片
            cv2.imshow(self.visualize_window_name, combined_img)
            
            # 按下q键退出可视化
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("🛑 用户按下q键，退出可视化")
                break
            
            rate.sleep()

    def run(self):
        """主运行流程"""
        try:
            # 步骤1：等待缓存准备就绪
            self.wait_for_cache_ready()
            
            # 步骤2：实时可视化数据
            self.visualize_data()
        
        except KeyboardInterrupt:
            rospy.loginfo("🛑 用户中断程序")
        finally:
            # 清理资源
            cv2.destroyAllWindows()
            rospy.loginfo("👋 程序正常退出")


if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node("cache_visualizer_node", anonymous=True)
    
    try:
        # 创建可视化器实例
        visualizer = CacheVisualizer(window_seconds=10.0, 
                                     img_w=630, 
                                     img_h=350)
        
        # 运行可视化流程
        visualizer.run()
    
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"❌ 程序异常：{e}")
        cv2.destroyAllWindows()
