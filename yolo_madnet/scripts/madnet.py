from __future__ import print_function

#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import message_filters
from sensor_msgs.msg import Image
from yolo_madnet.msg import DispMsg
from cv_bridge import CvBridge, CvBridgeError

import time
import sys, os
try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from madnet.Demo.demo_model import *


class madnet:
    """
    Initialize MADNet model with tensorflow and ROS.
    """
    def __init__(self):

        self.load_model()
        self.bridge = CvBridge()
        rospy.init_node('detection_node')
        sub_l = message_filters.Subscriber('/camera/infra1/image_rect_raw', Image)
        sub_r = message_filters.Subscriber('/camera/infra2/image_rect_raw', Image)
        ats = message_filters.ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.1)
        ats.registerCallback(self.process)
        self.pub = rospy.Publisher('/disparity', DispMsg, queue_size=10)
        self.msg_pub = DispMsg()
        rospy.spin()

    """
    ROS process. This function perform distance estimation with MADNet working
    through Tensorflow.

    See the GitHub of the author:
    https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo
    """
    def load_model(self):
        dd = RealTimeStereo(
            camera_frames,
            model_name='MADNet',
            weight_path='madnet/weights/MADNet/kitti',
            learning_rate=0.001,
            block_config_path='../block_config/MadNet_full.json',
            image_shape = [480,640],
            crop_shape=[320,512],
            SSIMTh = 0.5,
            mode = 'MAD'
            )
        dd.start()

    def process(self, msg_l, msg_r):
        # Input image from topic
        start = time.time()
        try:
            img_l = self.bridge.imgmsg_to_cv2(msg_l, "bgr8")
            img_r = self.bridge.imgmsg_to_cv2(msg_r, "bgr8")
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        madnet()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start distance estimation node.')
