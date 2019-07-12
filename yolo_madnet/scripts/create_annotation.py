import rospy
from cv_bridge import CvBridge, CvBridgeError
from yolo_madnet.msg import DetectionMsg

import cv2
import os, sys

class create_annotation:
    def __init__(self):
        rospy.init_node('create_annotation')
        self.bridge = CvBridge()
        object = rospy.Subscriber('/detection/image', DetectionMsg, self.process, queue_size=10)
        rospy.spin()

    def process(self, msg):
        # This function will create txt files associated to images with yolo format bbox
        #TODO

        # Convert ros image message to opencv images
        try:
            img = self.bridge.imgmsg_to_cv2(msg.image, "rgb8")
        except CvBridgeError as e:
            print(e)
        # We perform on every detected objects on the frame
        print('frame1')
        for object in msg.bbox:
            x1,y1,x2,y2 = object.x1, object.y1, object.x2, object.y2
            print((x1,y1), (x2,y2))

        return

if __name__ == '__main__':
    try:
        create_annotation()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start create_annotation node.')
