#! /usr/bin/python3

import rospy
from sensor_msgs.msg import Image
import cv2

from yolov3.detect import detect

class detection:
    """
    Initialize ROS
    """
    def __init__(self):

        rospy.init_node('detection_node')
        sub_l = rospy.Subscriber('/camera/infrared/left', Image, self.process, queue_size=10)
        pub = rospy.Publisher('/detection', BboxMsg, queue_size=10)

    """
    ROS process. This function perform object detection with Yolov3 working
    through Pytorch.

    See the GitHub of the author: https://github.com/ultralytics/yolov3
    """
    def process(self, msg):
        img = cv2.imread(msg)
        with torch.no_grad():
            detect(
                'cfg/yolov3-spp.cfg',
                'data/coco.data',
                'weights/yolov3-spp.weights',
                images=img,
                img_size=416,
                conf_thres=0.5,
                nms_thres=0.5,
                fourcc='mp4v',
                output='output'
            )

if __name__ == '__main__':
    try:
        detection()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start detection node.')
