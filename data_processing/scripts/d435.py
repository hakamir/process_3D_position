#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image


def d435():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
    pipe.start(cfg)
    rospy.init_node('d435_node')
    publ = rospy.Publisher('/camera/infrared/left', Image, queue_size=10)
    pubr = rospy.Publisher('/camera/infrared/right', Image, queue_size=10)

    #Create colorizer object
    colorizer = rs.colorizer();

    while not rospy.is_shutdown():
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        # Fetch pose frame
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)
        ir_left_color_frame = colorizer.colorize(ir_left_frame)
        ir_right_color_frame = colorizer.colorize(ir_right_frame)
        ir_left_color_image = np.asanyarray(ir_left_color_frame.get_data())
        ir_right_color_image = np.asanyarray(ir_right_color_frame.get_data())
        cv2.imshow("Infrared left Stream", ir_left_color_image)
        cv2.imshow("Infrared right Stream", ir_right_color_image)
        publ.publish(ir_left_color_image)
        pubr.publish(ir_right_color_image)
        

if __name__ == '__main__':
    try:
        d435()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')