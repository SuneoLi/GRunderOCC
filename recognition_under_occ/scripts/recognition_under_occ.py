#!/home/suneo/anaconda3/envs/ros/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy
import PIL.Image as PILImage 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

from img_reconstruction import strip2clean
from img_recognition import image2class


class RecognitionUnderOCC:
    def __init__(self):    
        # 创建cv_bridge，声明发布者和订阅者
        self.image_pub = rospy.Publisher("/model/image_reconstructed", Image, queue_size=1)
        self.command_pub = rospy.Publisher("/model/command", String, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_interfered", Image, self.callback)

    def callback(self, data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # opencv2pillow
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        # reconstruction
        pil_image_recon = strip2clean(pil_image)
        # pillow2opencv
        cv_image_recon = cv2.cvtColor(numpy.asarray(pil_image_recon), cv2.COLOR_RGB2BGR)

        # recognition
        cls_gesture = image2class(pil_image_recon)
        command_msg = String()
        command_msg.data = str(cls_gesture)

        # 显示Opencv格式的图像
        # cv2.imshow("Recon-Image window", cv_image_recon)
        # cv2.waitKey(3)

        # 再将opencv格式额数据转换成ros image格式的数据发布
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_recon, "bgr8"))

        self.command_pub.publish(command_msg)


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("recognition_under_occ")
        rospy.loginfo("Starting recognition_under_occ node")
        RecognitionUnderOCC()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
