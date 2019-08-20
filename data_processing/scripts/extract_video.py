#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Rodolphe Latour
"""

import cv2
import rospy
from sensor_msgs.msg import Image
from yolo_madnet.msg import DetectionMsg
from cv_bridge import CvBridge, CvBridgeError


class extract:
    """
    Description:
    ============
    The class give tools to show published images on ROS topic with OpenCV
    """
    def __init__(self):
        rospy.init_node('extract_video_node')
        self.bridge = CvBridge()
        rospy.Subscriber('/detection/image', DetectionMsg, self.process, queue_size=1)
        rospy.spin()

    def process(self, img):
        try:
            frame = self.bridge.imgmsg_to_cv2(img.image, "rgb8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        extract()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start post process node.')
