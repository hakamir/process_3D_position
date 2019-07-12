import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import argparse

import numpy as np
import cv2
import os, sys


def feed_img():
    parser=argparse.ArgumentParser(description='Feed images from path that will be publish in ros topic /feed/image')
    parser.add_argument("-p","--path", help='path to the images', default=None)
    args=parser.parse_args()
    rospy.init_node('feed_img_node')
    bridge = CvBridge()
    pub = rospy.Publisher('/feed/image', Image, queue_size=1)
    rate = rospy.Rate(5)
    for img in os.listdir(args.path):
        print(os.path.join(args.path,img))
        cv2_img = cv2.imread(os.path.join(args.path,img))
        print(cv2_img.shape)
        msg_img = bridge.cv2_to_imgmsg(cv2_img, "bgr8")
        pub.publish(msg_img)
        rate.sleep()




if __name__ == '__main__':
    try:
        feed_img()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start feed node.')
