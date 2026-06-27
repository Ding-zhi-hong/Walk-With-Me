#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus

def main():
    # 只初始化一次节点
    rospy.init_node("sim_odom_gps_publisher", anonymous=True)
    # 两个发布器
    pub_odom = rospy.Publisher("/odom", Odometry, queue_size=10)
    pub_gps = rospy.Publisher("/fix", NavSatFix, queue_size=10)
    r = rospy.Rate(10)  # 统一10Hz

    # ========== Odometry 消息初始化 ==========
    odom_msg = Odometry()
    odom_msg.header.frame_id = "odom"
    odom_msg.child_frame_id = "base_link"
    odom_msg.pose.pose.orientation.w = 1.0
    odom_msg.twist.twist.linear.x = 0.2
    odom_msg.twist.twist.angular.z = 0.1
    odom_msg.pose.covariance = [0.0] * 36
    odom_msg.twist.covariance = [0.0] * 36
    pos_x = 0.0
    step = 0.1

    # ========== NavSatFix GPS消息初始化 ==========
    gps_msg = NavSatFix()
    gps_msg.header.frame_id = "gps_link"
    gps_msg.status.status = NavSatStatus.STATUS_FIX
    gps_msg.status.service = NavSatStatus.SERVICE_GPS
    gps_msg.latitude = 31.820191479442805
    gps_msg.longitude = 117.12819066395285
    gps_msg.altitude = 0.0
    gps_msg.position_covariance = [0.0] * 9
    gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    rospy.loginfo("同时发布 /odom 和 /gps/fix 话题，10Hz")
    while not rospy.is_shutdown():
        now = rospy.Time.now()

        # 更新并发布里程计
        odom_msg.header.stamp = now
        odom_msg.pose.pose.position.x = pos_x
        pub_odom.publish(odom_msg)
        pos_x += step
        step += 0.1

        # 更新并发布GPS fix
        gps_msg.header.stamp = now
        pub_gps.publish(gps_msg)

        r.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点退出")
