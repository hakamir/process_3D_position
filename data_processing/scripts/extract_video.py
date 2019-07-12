import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class extract:
    def __init__(self):
        rospy.init_node('extract_node')
        self.bridge = CvBridge()
        rospy.Subscriber('/object/detected', Image, self.process, queue_size=1)
        rospy.spin()

    def process(self, img):
        print('ok')
        try:
            frame = self.bridge.imgmsg_to_cv2(img, "rgb8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        extract()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start post process node.')
