#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""

import rospy
from data_processing.msg import CameraMsg

"""
This script transmits the pose data of the RealSense T265. It sends a Vector3
for the linear position and a Quaternion for the angular position.
"""

def t265():

    rospy.init_node('t265_node')
    pub = rospy.Publisher('/camera/position', CameraMsg, queue_size=10)
    rate = rospy.Rate(10) # Select the frame rate you want here.
    msg = CameraMsg()
    while not rospy.is_shutdown():

    	x = 0.12
    	y = 0.01
    	z = 0.04
    	rx = 0.02
    	ry = 0.001
    	rz = 0.014
    	rw = 0.991

    	now = rospy.get_rostime()
    	msg.header.stamp.secs = now.secs
    	msg.header.stamp.nsecs = now.nsecs
    	msg.linear.x = x
    	msg.linear.y = y
    	msg.linear.z = z
    	msg.angular.x = rx # complex i
    	msg.angular.y = ry # complex j
    	msg.angular.z = rz # complex k
    	msg.angular.w = rw # scalar w

    	pub.publish(msg)
    	rate.sleep()


if __name__ == '__main__':
    try:
        t265()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')
