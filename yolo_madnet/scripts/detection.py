from __future__ import print_function

#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from yolo_madnet.msg import BboxMsg
from cv_bridge import CvBridge, CvBridgeError

import time
import sys, os
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

        self.load_model()
        self.bridge = CvBridge()
        rospy.init_node('detection_node')
        #sub_l = rospy.Subscriber('/usb_cam/image_raw', Image, self.process, queue_size=10)
        sub_l = rospy.Subscriber('/camera/color/image_raw', Image, self.process, queue_size=10)
        self.pub = rospy.Publisher('/detection', BboxMsg, queue_size=10)
        self.msg_pub = BboxMsg()
        rospy.spin()

    """
    ROS process. This function perform object detection with Yolov3 working
    through Pytorch.

    See the GitHub of the author: https://github.com/ultralytics/yolov3
    """
    def load_model(self):
        # Resize from yolo model and set confidence and iou thresholds
        self.img_size = 608
        self.conf_thres = 0.5
        self.nms_thres = 0.5
        # Path to config and weights files
        cfg = 'yolov3/cfg/yolov3-spp.cfg'
        weights = 'yolov3/weights/yolov3-spp.weights'
        class_names = 'yolov3/data/coco.names'

        # Load model and weights
        print('Loading models...')
        # Initialize model
        if ONNX_EXPORT:
            s = (320, 192)  # onnx model image size (height, width)
            self.model = Darknet(cfg, s)
        else:
            self.model = Darknet(cfg, self.img_size)
        _ = load_darknet_weights(self.model, weights)
        # Fuse Conv2d + BatchNorm2d layers
        self.model.fuse()

        # Eval mode
        self.device = torch_utils.select_device()
        self.model.to(self.device).eval()
        # Get classes and colors
        self.classes = load_classes(class_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def process(self, msg):
        # Input image from topic
        #os.system('clear')
        start = time.time()
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            orig_img = img
            if ONNX_EXPORT:
                img = torch.zeros((1, 3, s[0], s[1]))
                torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
                return
        except CvBridgeError as e:
            print(e)
        # Padded resize
        img, _, _, _ = self.letterbox(orig_img)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img.shape[:2]).round()
            for item in range(len(det)):
                label = '%s' % (self.classes[int(det[item][6].unique())])
                score = det[item][4]
                pos = det[item][0:4].tolist()
                for i in range(len(pos)):
                    pos[i] = int(pos[i])
                x1, y1, x2, y2 = pos
                now = rospy.get_rostime()
                self.msg_pub.header.stamp.secs = now.secs
                self.msg_pub.header.stamp.nsecs = now.nsecs
                self.msg_pub.x1 = x1
                self.msg_pub.y1 = y1
                self.msg_pub.x2 = x2
                self.msg_pub.y2 = y2
                self.msg_pub.obj_class = label
                self.msg_pub.score = score
                self.pub.publish(self.msg_pub)
                #print('Detect {} with {} score at {}, {} position'.format(label, round(score,2), (x1, y1), (x2, y2)))
                plot_one_box(pos, orig_img, label=label, color=self.colors[int(det[item][6])])
        print('FPS: {}'.format(int(1 / (time.time() - start))))
        cv2.imshow('Detection', orig_img)
        cv2.waitKey(1)


    def letterbox(self, image, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            ratio = float(new_shape) / max(shape)
        else:
            ratio = max(new_shape) / max(shape)  # ratio  = new / old
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0]) / 2  # width padding
            dh = (new_shape - new_unpad[1]) / 2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        return image, ratio, dw, dh



if __name__ == '__main__':
    try:
        detection()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start detection node.')
