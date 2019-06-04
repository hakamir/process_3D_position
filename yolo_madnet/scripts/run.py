#! /usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from yolo_madnet.msg import BboxMsg
from cv_bridge import CvBridge, CvBridgeError

import sys
try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from yolov3.models import *
from yolov3.utils.utils import *


class detection:
    """
    Initialize Yolov3 model with Pytorch and ROS.
    """
    def __init__(self):

        # Resize from yolo model and set confidence and iou thresholds
        img_size = 416 # You can set : 320, 416 and 608
        conf_thres = 0.5
        nms_thres = 0.5
        # Path to config and weights files
        cfg = 'yolov3/cfg/yolov3.cfg'
        weights = 'yolov3/weights/yolov3.weights'
        data_cfg = 'yolov3/data/coco.data'
        class_names = 'yolov3/data/coco.names'

        # Load model and weights
        print('Loading models...')
        try:
            self.model = Darknet(cfg, img_size)
            _ = load_darknet_weights(self.model, weights)
        except:
            raise('ERROR Cannot load models. Segmentation fault (core dumped)')
        # Fuse Conv2d + BatchNorm2d layers
        self.model.fuse()

        # Eval mode
        device = torch_utils.select_device()
        self.model.to(device).eval()
        # Get classes and colors
        classes = load_classes(class_names)
        #classes = load_classes(data_cfg)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        rospy.init_node('detection_node')
        sub_l = rospy.Subscriber('/camera/infra1/image_rect_raw', Image, self.process, queue_size=10)
        pub = rospy.Publisher('/detection', BboxMsg, queue_size=10)
        rospy.spin()

    """
    ROS process. This function perform object detection with Yolov3 working
    through Pytorch.

    See the GitHub of the author: https://github.com/ultralytics/yolov3
    """
    def process(self, msg):
        # Input image from topic
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            print(det)


if __name__ == '__main__':
    try:
        detection()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start detection node.')
