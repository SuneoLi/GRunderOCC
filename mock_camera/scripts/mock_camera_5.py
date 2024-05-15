#!/home/suneo/anaconda3/envs/ros/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import faulthandler
faulthandler.enable()


def mock_camera():
    # ROS节点初始化
    rospy.init_node('mock_camera', anonymous=True)    

    # 创建一个Publisher，发布名为/camera/image_interfered的topic，消息类型为sensor_msgs::Image，队列长度1
    image_pub = rospy.Publisher('/camera/image_interfered', Image, queue_size=1)

    #设置循环的频率
    rate = rospy.Rate(10) 

    while not rospy.is_shutdown():
        # 初始化opencv格式的消息
        cv_image = cv2.imread('/home/suneo/catkin_ws/src/mock_camera/resource/5.jpg')

	# 发布消息 将opencv格式的数据转换成rosimage格式的数据发布
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        image_pub.publish(ros_image)

        rospy.loginfo("Publishing image...")

	# 按照循环频率延时
        rate.sleep()


if __name__ == '__main__':
    try:
        mock_camera()
    except rospy.ROSInterruptException:
        pass


