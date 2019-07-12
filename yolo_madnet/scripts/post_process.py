#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import message_filters
from sensor_msgs.msg import Image
from yolo_madnet.msg import DetectionMsg
from data_processing.msg import PointMsg
from data_processing.msg import PointsMsg
from cv_bridge import CvBridge, CvBridgeError

import cv2
"""
try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
"""
import random

import numpy as np
import time

from sort.utils.object_segmentation import segmentation
#from sort.sort_3d import Sort

class post_process:
    """
    This script manage the post-process managing the assimilation between the
    disparity map and the bounding boxes.
    Input:
    - disparity map from topic /disparity
    - bounding boxes from topic /detection/bbox
    - camera image with detection from topic /detection/image
    Output:
    - post-processed position in (x,y,z) to topic /object/detected
    - masked image with distance value to topic /detection/distance

    It currently doesn't provide the annonced output. Some new function will be
    implemented. The spacialize node will be merged with this one.
    """
    def __init__(self):
        # D435 parameters
        self.baseline = 49.867050e-3
        self.focal = 1.93e-3
        self.pixel_size = 3e-6

        # Loading class names file and generate random colors for bounding boxes
        self.classes = self.load_classes('adapt/adapt.names')
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        self.bridge = CvBridge()

        # Init ROS node, publishers and subscribers
        rospy.init_node('post_process_node')
        disparity = message_filters.Subscriber('/disparity', Image)
        object = message_filters.Subscriber('/detection/image', DetectionMsg)
        ats = message_filters.ApproximateTimeSynchronizer([disparity, object], queue_size=1, slop=0.1)
        ats.registerCallback(self.process)
        self.pub = rospy.Publisher('/object/detected', PointsMsg, queue_size=0)
        self.pub_img = rospy.Publisher('/detection/distance', Image, queue_size=0)
        rospy.spin()

    """
    This is the function run everytime the three messages in input are received.
    It manages all the process to provide the asked output topics.
    """
    def process(self, disp, obj):
        points_list = PointsMsg()
        # Convert ros image message to opencv images
        try:
            disparity = self.bridge.imgmsg_to_cv2(disp, "32FC1")
            img = self.bridge.imgmsg_to_cv2(obj.image, "rgb8")
        except CvBridgeError as e:
            print(e)
        # We perform on every detected objects on the frame
        for object in obj.bbox:
            x1,y1,x2,y2 = object.x1, object.y1, object.x2, object.y2
            bbox = [x1, y1, x2, y2]
            # We take a mask of the object from the disparity to improve results
            mask, img = segmentation(bbox, disparity, img)

            # We get the distance in meter
            # It correspond to the direct distance from the focal point, not z
            distance = self.disp_mask(mask[0][0], disparity, bbox)
            #distance = self.get_object_distance(mask, disparity, bbox,
            #            focal_lenght=1.93, baseline=49.867050, pixel_size=3)

            # We define the position of the object on the image at the center of the box
            #distance = ((2.0/3.0)*self.focal*self.baseline)/(self.pixel_size*disparity[(x2 - x1)/2][(y2 - y1)/2]/255)
            u = (x2 - x1) / 2 + x1 - img.shape[1]/2
            v = (y2 - y1) / 2 + y1 - img.shape[0]/2

            # We found (x,y,z) positions through those equations
            factor = np.sqrt(1/(self.focal**2 + self.pixel_size**2 * u**2 + self.pixel_size**2 * v**2))
            X = distance * self.pixel_size * u * factor
            Y = distance * self.pixel_size * v * factor
            Z = distance * self.focal * factor

            # Calculate the scale of the bounding box that will be transposed in 3D
            x_1 = u - (x2 - x1) / 2
            x_2 = u + (x2 - x1) / 2
            y_1 = v - (y2 - y1) / 2
            y_2 = v + (y2 - y1) / 2
            factor = np.sqrt(1/(self.focal**2 + self.pixel_size**2 * (x_2 - x_1)**2 + self.pixel_size**2 * (y_2 - y_1)**2))
            a = distance * self.pixel_size * (x_2 - x_1) * factor
            b = distance * self.pixel_size * (y_2 - y_1) * factor
            c = np.sqrt(a*b)
            scale = [a, b, c]

            # Now append data to the message
            _class = object.obj_class
            score = object.score
            now = rospy.get_rostime()
            point = PointMsg()
            point.header.stamp.secs = now.secs
            point.header.stamp.nsecs = now.nsecs
            point.position.x = X
            point.position.y = Y
            point.position.z = Z
            point.scale.x = scale[0]
            point.scale.y = scale[1]
            point.scale.z = scale[2]
            point.obj_class = _class
            point.score = score
            points_list.point.append(point)

            # Plot box to the image for visualisation
            label = _class + " " + str(round(distance,2)) + ' m'
            for cls in self.classes:
                if _class == cls:
                    color_index = self.classes.index(cls)
            self.plot_one_box(bbox, img, label=label, color=self.colors[color_index])

        # Publish the visualisation of bounding box with distance and mask
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))
        # Publish the message
        now = rospy.get_rostime()
        points_list.header.stamp.secs = now.secs
        points_list.header.stamp.nsecs = now.nsecs
        self.pub.publish(points_list)
        print(points_list)

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    """
    We use this function to plot boxes on the image
    """
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
            if c1[1] > 30:
                cv2.rectangle(img, c1, c2, color, -1)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=int(tf), lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(img, (c1[0], c1[1] + 30), (c2[0], c2[1] - 2 + 30), color, -1)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2 + 30), 0, tl / 3, [225, 255, 255], thickness=int(tf), lineType=cv2.LINE_AA)

    """
    This function search the median point of the disparity map from the masked
    object. It returns the distance in meter.
    """
    def disp_mask(self, mask, disp, bbox, focal_lenght=1.93e-3, baseline=49.867050e-3, pixel_size=3e-6):
        x1, y1, x2, y2 = bbox
        # We crop the disparity map image to the bounding box position
        disp_seg=disp[y1:y2,x1:x2]/255
        # We keep only the disparity values of the bounding box that aren't masked
        disp_seg = np.ma.masked_array(disp_seg, mask=mask)
        median = np.median(disp_seg.data)
        # We now calculate the distance in meter
        if median==0:
            median= ((2.0/3.0) * focal_lenght * baseline) / (pixel_size*(median + 0.001))
        else:
            median= ((2.0/3.0) * focal_lenght * baseline) / (pixel_size*median)
        return median

    """
    This function is deprecated
    """
    def get_object_distance(self, mask, disp, bbox, focal_lenght, baseline, pixel_size):
        x1, y1, x2, y2 = bbox
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
