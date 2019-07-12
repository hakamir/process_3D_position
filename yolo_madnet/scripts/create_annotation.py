import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import os, sys

class create_annotation:
    def __init__(self):
        rospy.init_node('create_annotation')
        self.bridge = CvBridge()
        object = rospy.Subscriber('/detection/image', DetectionMsg, self.process, queue_size=10)
        rospy.spin()

    def process(self):
        # This function will create txt files associated to images with yolo format bbox
        #TODO
        return

if __name__ == '__main__':
    try:
        create_annotation()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start create_annotation node.')
