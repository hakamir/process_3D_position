#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""
import pyrealsense2 as rs
import rospy
from data_processing.msg import CameraMsg

"""
This script transmits the pose data of the RealSense T265. It sends a Vector3
for the linear position and a Quaternion for the angular position. 
"""

def t265():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    rospy.init_node('t265_node')
    pub = rospy.Publisher('/camera/position', CameraMsg, queue_size=10)
    rate = rospy.Rate(10) # Select the frame rate you want here.
    msg = CameraMsg()
    while not rospy.is_shutdown():
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        # Fetch pose frame
        pose = frames.get_pose_frame()
        if pose:
            # Print some of the pose data to the terminal
            data = pose.get_pose_data()
            now = rospy.get_rostime()
            msg.header.stamp.secs = now.secs
            msg.header.stamp.nsecs = now.nsecs
            msg.linear.x = data.translation.x
            msg.linear.y = data.translation.y
            msg.linear.z = data.translation.z
            msg.angular.x = data.rotation.x # complex i
            msg.angular.y = data.rotation.y # complex j
            msg.angular.z = data.rotation.z # complex k
            msg.angular.w = data.rotation.w # scalar w
            
            pub.publish(msg)
            rate.sleep()
        

if __name__ == '__main__':
    try:
        t265()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')