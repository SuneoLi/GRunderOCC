#!/home/suneo/anaconda3/envs/ros/bin/python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class TurtleController:
    def __init__(self):    
        # 声明发布者和订阅者
        self.cmd_pub = rospy.Publisher("/turtle1/cmd_vel", Twist, queue_size=10)
        self.cmd_sub = rospy.Subscriber("/model/command", String, self.callback)

    def callback(self, data):
        # 
        cls_gesture = str(data.data)
        # print(cls_gesture)

        if cls_gesture == '0':
            vel_msg = Twist()
        elif cls_gesture == '1':
            vel_msg = Twist()
            vel_msg.linear.x = 0.5
        elif cls_gesture == '2':
            vel_msg = Twist()
            vel_msg.linear.x = -0.5
        elif cls_gesture == '3':
            vel_msg = Twist()
            vel_msg.linear.y = 0.5
        elif cls_gesture == '4':
            vel_msg = Twist()
            vel_msg.linear.y = -0.5
        elif cls_gesture == '5':
            vel_msg = Twist()
            vel_msg.angular.z = 0.2
        else:
            rospy.loginfo("ERROR")

        # 发布
        self.cmd_pub.publish(vel_msg)


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("turtle_controller")
        rospy.loginfo("Starting turtle_controller node")
        TurtleController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
