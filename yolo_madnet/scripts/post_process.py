#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import message_filters
from sensor_msgs.msg import Image
from yolo_madnet.msg import BboxMsg
from data_processing.msg import PostObjectMsg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import random

import numpy as np
import time

class post_process:
    def __init__(self):
        print('Initialize node...')
        self.classes = self.load_classes('yolov3/data/coco.names')
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        self.bridge = CvBridge()
        rospy.init_node('post_process_node')
        disparity = message_filters.Subscriber('/disparity', Image)
        object = message_filters.Subscriber('/detection/bbox', BboxMsg)
        color = message_filters.Subscriber('/detection/image', Image)
        ats = message_filters.ApproximateTimeSynchronizer([disparity, object, color], queue_size=3, slop=0.1)
        ats.registerCallback(self.process)
        self.pub = rospy.Publisher('/object/post', PostObjectMsg, queue_size=0)
        self.pub_img = rospy.Publisher('/detection/distance', Image, queue_size=0)
        self.msg_pub = PostObjectMsg()
        rospy.spin()

    def process(self, disp, obj, color):
        try:
            disparity = self.bridge.imgmsg_to_cv2(disp, "16UC1")
            img = self.bridge.imgmsg_to_cv2(color, "rgb8")
        except CvBridgeError as e:
            print(e)
        out_list = []
        x1,y1,x2,y2 = obj.x1, obj.y1, obj.x2, obj.y2
        _class = obj.obj_class
        score = obj.score
        distance = self.get_object_distance(disparity, x1, y1, x2, y2,
                        focal_lenght=1.93, baseline=49.867050, pixel_size=3)
        u = (x2 - x1) / 2
        v = (y2 - y1) / 2
        self.msg_pub.position.x = u
        self.msg_pub.position.y = v
        self.msg_pub.position.z = distance
        self.msg_pub.obj_class = _class
        self.msg_pub.score = score
        print(self.msg_pub)
        self.pub.publish(self.msg_pub)
        pos = [x1, y1, x2, y2]
        label = _class + " " + str(round(distance,2)) + ' m'
        for cls in self.classes:
            if _class == cls:
                color_index = self.classes.index(cls)
        self.plot_one_box(pos, img, label=label, color=self.colors[color_index])
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=int(tl))
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=int(tf))[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=int(tf), lineType=cv2.LINE_AA)

    def get_object_distance(self,disp,x1,y1,x2,y2,focal_lenght,baseline,pixel_size):
	    hist_with_indx=np.zeros((65536,2))
	    hist_with_indx[:,0]=np.arange(65536)
	    x1_prim=x1+9/10*(x2-x1)/2
	    y1_prim=y1+9/10*(y2-y1)/2
	    x2_prim=x2-9/10*(x2-x1)/2
	    y2_prim=y2-9/10*(y2-y1)/2
	    histr = cv2.calcHist([disp[int(y1_prim):int(y2_prim),int(x1_prim):int(x2_prim)]],[0],None,[65536],[0,65535])
	    distance_list=[]
	    hist_with_indx[:,1]=histr[:,0]
	    indx_to_remove=np.argmax(hist_with_indx[:,1])
	    distance=int(hist_with_indx[indx_to_remove,0])/255
	    max_hist=hist_with_indx[indx_to_remove,1]
	    hist_with_indx=np.delete(hist_with_indx,(indx_to_remove),axis=0)
	    max_score=max_hist*indx_to_remove**2
	    if distance==0:
	        distance=focal_lenght * baseline / (pixel_size*(distance + 0.001)) #Pour KITTI
	    else:
	        distance=focal_lenght * baseline / (pixel_size*distance)
	    distance_list.append(distance)
	    for i in range(3):
	        indx_to_remove=np.argmax(hist_with_indx[:,1])
	        score=hist_with_indx[indx_to_remove,1]*hist_with_indx[indx_to_remove,0]**2
	        if score>max_score:
	            max_score=score

	        distance=int(hist_with_indx[indx_to_remove,0])/255
	        ecart=abs(score-max_score)
	        quadra=ecart/max_score
	        if quadra<0.1:
	            if distance==0:
	                distance=focal_lenght * baseline / (pixel_size*(distance + 0.001))
	            else:
	                distance=focal_lenght * baseline / (pixel_size*distance)
	            distance_list.append(distance)
	        hist_with_indx=np.delete(hist_with_indx,(indx_to_remove),axis=0)
	    distance=np.sum(distance_list)/len(distance_list)
	    return distance

if __name__ == '__main__':
    try:
        post_process()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start post process node.')
