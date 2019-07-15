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
        self.i = 0
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
        size = img.shape
        img_name = 'img_%08d.jpg'%self.i
        txt_name = 'img_%08d.txt'%self.i
        f = open('annotation/' + txt_name, 'a')
        for object in msg.bbox:
            bbox = object.x1, object.y1, object.x2, object.y2
            x, y, w, h = self.convert(size, bbox)
            print(object.obj_class)
            if object.obj_class == 'door':
                _class = 0
            elif object.obj_class == 'handle':
                _class = 1
            elif object.obj_class == 'elevator':
                _class = 2
            elif object.obj_class == 'switch':
                _class = 3
            print(_class, x, y, w, h)
            f.write('{} {} {} {} {}\n'.format(_class, x, y, w, h))
        cv2.imwrite('annotation/' + img_name, img)
        self.i += 1
        return

    def convert(self,size, box):
        dw = 1./size[1]
        dh = 1./size[0]
        x = (box[0] + box[2])/2.0
        y = (box[1] + box[3])/2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

if __name__ == '__main__':
    try:
        create_annotation()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start create_annotation node.')
