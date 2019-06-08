#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""
import rospy
from data_processing.msg import ObjectMsg

"""
This script is used to simulate the yolov3 and MADNet detector and estimator
passing through a point convector. Respectively : YoloMADNet and Position3D nodes. 

It gives spacial coordinates of a point depending the referential
of the camera (not the global referential so), the class of the detected object
and the score of the detection. 
"""

def obj_sim(x_factor, y_factor, z_factor):
    rospy.init_node('obj_sim__node')
    pub = rospy.Publisher('/object', ObjectMsg, queue_size=10)
    rate = rospy.Rate(10)
    msg = ObjectMsg()
    x = 0
    y = 0
    z = 2
    way = 0
    while not rospy.is_shutdown():
        
        if way == 0:
            x += x_factor
            if x >= 10.0:
                way = 1
        elif way == 1:
            x -= x_factor
            if x <= 1.0:
                way = 0

        
        now = rospy.get_rostime()
        msg.header.stamp.secs = now.secs
        msg.header.stamp.nsecs = now.nsecs
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.obj_class = 'person'
        msg.score = 0.80
        pub.publish(msg)
        rate.sleep()
        
if __name__ == '__main__':
    try:
        obj_sim(0.0,0.0,0.0)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')
