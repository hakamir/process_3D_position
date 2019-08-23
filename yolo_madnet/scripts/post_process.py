#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Rodolphe Latour
"""

import rospy
import message_filters
from sensor_msgs.msg import Image
from yolo_madnet.msg import DetectionMsg
from yolo_madnet.msg import PointMsg
from yolo_madnet.msg import PointsMsg
from cv_bridge import CvBridge, CvBridgeError
import argparse

#import cv2

# If the ROS path cause trouble, uncomment.
try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import random

import numpy as np
import time

from sort.utils.object_segmentation import segmentation
from sort.sort_3d import Sort

class post_process:
    """
    Description:
    ============
    This script manage the post-process taking care of the assimilation between
    the disparity map and the bounding boxes.

    Input:
    ------
    - disparity map from topic /disparity
    - bounding boxes from topic /detection/bbox
    - camera image with detection from topic /detection/image

    Output:
    -------
    - post-processed position in (x,y,z) to topic /object/detected
    - masked image with distance value to topic /detection/distance
    """
    def __init__(self):
        # D435 parameters (set yours if different)
        self.baseline = 49.867050e-3
        self.focal = 1.93e-3
        self.pixel_size = 3e-6
        self.cell_width = 1280
        self.cell_height = 720

        self.mot_tracker = Sort()

        parser=argparse.ArgumentParser(description='Script for post processing object detection and distance estimation.')
        parser.add_argument("-m","--model", help='Paths to the different given dataset (adapt, coco)', default="adapt")
        args=parser.parse_args(rospy.myargv()[1:])
        # Loading class names file and generate random colors for bounding boxes
        if args.model == 'adapt':
            self.class_file_path = 'adapt/adapt.names' # Door, handle, elevator and switch
        elif args.model == 'coco':
            self.class_file_path = 'yolov3/data/coco.names' # COCO dataset classes
        else:
            raise NameError("'{}' file is not accessible in the script.".format(self.class_file_path))
        self.classes = self.load_classes(self.class_file_path)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        self.bridge = CvBridge()

        # Init ROS node, publishers and subscribers
        rospy.init_node('post_process_node')
        disparity = message_filters.Subscriber('/disparity', Image)
        object = message_filters.Subscriber('/detection/image', DetectionMsg)
        # We use a time synchronizer to get data from MADNet and Yolov3 at the same time
        ats = message_filters.ApproximateTimeSynchronizer([disparity, object], queue_size=1, slop=0.1)
        # The process method will run when both data will be acquired
        ats.registerCallback(self.process)
        self.pub = rospy.Publisher('/object/detected', PointsMsg, queue_size=0)
        self.pub_img = rospy.Publisher('/detection/distance', Image, queue_size=0)
        rospy.spin()



    def process(self, disp, obj):
        """
        Description:
        ============
        This is the function run everytime the three messages in input are
        received. It manages all the process to provide the asked output topics.

        Input:
        ------
        - disp: The disparity map from MADNet node (named madnet.py)
        - obj: The list of object per image and the associated image from
        yolov3 node (named detection.py)

        Output:
        -------
        - img: The images with the detected objects surrounded with bounding
        boxes and a label with the class name and the relative distance.
        - points_list: A list of all detected object. Each element is built
        that way:
            * Vector3 position
             -- x: the x position in space (relative to the camera position)
             -- y: the y position in space (relative to the camera position)
             -- z: the z position in space (relative to the camera position)

            * Vector3 scale
             -- scale_x: the scale in x of the box based on the bounding box
             -- scale_y: the scale in y of the box based on the bounding box
             -- scale_z: the scale in z of the box based on the bounding box

            * string obj_class: The class of the object

            * float32 score: the detection score of the object
        """

        # initialize variable for sort
        self.distances = []
        self.dets = []

        points_list = PointsMsg()
        # Convert ROS image message to opencv images
        try:
            # For the disparity, the type is 32FC1
            disparity = self.bridge.imgmsg_to_cv2(disp, "32FC1")
            # For the raw image (left camera), the type is the standard RGB8
            img = self.bridge.imgmsg_to_cv2(obj.image, "rgb8")
        except CvBridgeError as e:
            print(e)
        # We perform on every detected objects on the frame
        for object in obj.bbox:
            # We set bounding box information in a list 'bbox'
            x1,y1,x2,y2 = object.x1, object.y1, object.x2, object.y2
            bbox = [x1, y1, x2, y2]
            # We get the size of the provided image (it must me resized)
            self.img_width = img.shape[1]
            self.img_height = img.shape[0]

            # We take a mask of the object from the disparity to improve results
            # The mask is usefull to avoid taking account of background
            #mask, img = segmentation(bbox, disparity, img)

            # We get the distance in meter
            # It correspond to the direct distance from the focal point, not z
            #distance = self.disp_mask(mask[0][0], disparity, bbox)
            distance = self.median_calculation(disparity, bbox)

            # Performing tracking with SORT
            tmp = bbox
            tmp.append(object.score)
            # Append data to track
            self.dets.append(tmp)
            self.distances.append(distance)

            # Perform tracking
            trackers,predictions,predicted_state,state_x = np.array(self.mot_tracker.update(np.array(self.dets),np.array(self.distances))[0])

            # concatenate every previous results
            if len(trackers)>0:
                trackers=np.concatenate(trackers)
            if len(predicted_state)>0:
                predicted_state=np.concatenate(predicted_state)
            if len(predictions)>0:
                predictions=np.concatenate(predictions)
            if len(state_x)>0:
                state_x=np.concatenate(state_x)

            try:
                print(trackers[0])
                bbox = trackers[1]
                x1, y1, x2, y2, ID = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            except:
                continue

            # We define the position of the object on the image at the center of the box
            u = (x2 - x1) / 2 + x1 - self.img_width/2
            v = (y2 - y1) / 2 + y1 - self.img_height/2

            # We found (x,y,z) object positions from camera referential
            # through those equations.
            factor = np.sqrt(1/(self.focal**2 + self.pixel_size**2 * u**2 + self.pixel_size**2 * v**2))
            X = distance * self.pixel_size * u * factor
            Y = distance * self.pixel_size * v * factor
            Z = distance * self.focal * factor

            # Calculate the scale of the 3D 'bounding' box
            x_1 = u - (x2 - x1) / 2
            x_2 = u + (x2 - x1) / 2
            y_1 = v - (y2 - y1) / 2
            y_2 = v + (y2 - y1) / 2
            factor = np.sqrt(1/(self.focal**2 + self.pixel_size**2 * (x_2 - x_1)**2 + self.pixel_size**2 * (y_2 - y_1)**2))
            a = distance * self.pixel_size * (x_2 - x_1) * factor
            b = distance * self.pixel_size * (y_2 - y_1) * factor
            # We arbitrary say that the depth of the box (z) is equal to the root of a*b
            c = np.sqrt(a*b)
            scale = [a, b, c]

            # Preparing stamptime for header and message object
            _class = object.obj_class
            score = object.score
            now = rospy.get_rostime()
            point = PointMsg()
            # Now append data to the new message
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
            # Then add each message to a list (one message per object)
            """
            rospy.loginfo('OBJECT: {}\n SCORE: {}\n POSITION:({}, {}, {})'.format(
                    point.obj_class,
                    point.score,
                    X,
                    point.position.y,
                    point.position.z))
            """
            points_list.point.append(point)

            # Plot box to the image for visualisation
            label = _class + " " + str(round(distance,2)) + ' m'
            # We set a color depending the index of the class in the list
            color_index = self.classes.index(_class)
            # Then, we plot a box on the raw image. One by detected object.
            self.plot_one_box(bbox, img, label=label, color=self.colors[color_index])

        # Publish the visualisation of bounding box with distance and mask
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))
        # Publish the message
        now = rospy.get_rostime()
        points_list.header.stamp.secs = now.secs
        points_list.header.stamp.nsecs = now.nsecs
        self.pub.publish(points_list)



    def load_classes(self, path):
        """
        Description:
        ============
        A function to load all the classes attributed to a trained model that
        are accessible in a list. The method take care to remove empty strings.
        The files to load end with '*.names'.

        Input:
        ------
        - path: The path of the text file containing the classes (one by line)

        Output:
        - list: A list containing for each element a class from the model
        """
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)



    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        """
        Description:
        ============
        This function plot boxes on the image with specific color.
        A rectangle is drawn around the object and on the top the class of the
        object and the associated distance is written.

        Input:
        ------
        - x: The bounding box of the detected object (x1, y1, x2, y2) in a list type
        - img: The input raw image where the box will be drawn
        - color: The color of the box and the text (optional)
        - label: The containt of the label on the top (optional)
        - line_thickness: The thickness of the rectangle (optional)
        """
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



    def disp_mask(self, mask, disp, bbox, focal_lenght=1.93e-3, baseline=49.867050e-3, pixel_size=3e-6):
        """
        DEPRECATED
        --> The median calculation doesn't take care of the mask

        Description:
        ============
        This function search the median point of the disparity map from the masked
        object. It returns the distance in meter.

        Input:
        ------
        - mask: The mask of the object on the bounding box based on Otsu method
        - disp: The disparity map
        - bbox: The bounding box position of the detected objects
        - focal_lenght: The focal lenght of the used camera (default:1.93e-3)
        - baseline: The baseline of the setup used (default:49.867050e-3)
        - pixel_size: The size of a pixel in the cell of the used camera (default:3e-6)

        Output:
        -------
        - median: The distance in meter of the object based on the median of each
        disparity value given in the input mask
        """
        x1, y1, x2, y2 = bbox
        # We crop the disparity map image to the bounding box position
        disp_seg=disp[y1:y2,x1:x2]/255
        # We keep only the disparity values of the bounding box that aren't masked
        disp_seg = np.ma.masked_array(disp_seg, mask=mask)
        median = np.median(disp_seg.data)
        # We now calculate the distance in meter with the following equation:
        # distance = resize * (focal * baseline)/(pixel_size * disparity)+
        if median==0:
            median = ((float(self.img_width)/float(self.cell_width)) * focal_lenght * baseline) / (pixel_size*(median + 0.001))
        else:
            median = ((float(self.img_width)/float(self.cell_width)) * focal_lenght * baseline) / (pixel_size*median)
        return median



    def median_calculation(self, disp, bbox, focal_lenght=1.93e-3, baseline=49.867050e-3, pixel_size=3e-6):
        """
        Description:
        ============
        This function search the median point of the disparity map without
        using the mask method. It returns the distance in meter.

        Input:
        ------
        - disp: The disparity map
        - bbox: The bounding box position of the detected objects
        - focal_lenght: The focal lenght of the used camera (default:1.93e-3)
        - baseline: The baseline of the setup used (default:49.867050e-3)
        - pixel_size: The size of a pixel in the cell of the used camera (default:3e-6)

        Output:
        -------
        - median: The distance in meter of the object based on the median of each
        disparity value.
        """
        x1, y1, x2, y2 = bbox
        # We crop the disparity map image to the bounding box position
        disp_seg=disp[y1:y2,x1:x2]/255
        # We calculate the median of the disparity points
        median = np.median(disp_seg)
        # We now calculate the distance in meter with the following equation:
        # distance = resize * (focal * baseline)/(pixel_size * disparity)
        if median==0:
            median = ((float(self.img_width)/float(self.cell_width)) * focal_lenght * baseline) / (pixel_size*(median + 0.001))
        else:
            median = ((float(self.img_width)/float(self.cell_width)) * focal_lenght * baseline) / (pixel_size*median)
        return median



    def get_object_distance(self, mask, disp, bbox, focal_lenght=1.93e-3, baseline=49.867050e-3, pixel_size=3e-6):
        """
        !!!DEPRECATED!!!

        /!\ The function was provided uncommented /!\

        Description:
        ============
        The method gives the distance of the center of the bounding box after
        the mask application. Most usefull for plane object but doesn't take in
        account object in front of it.

        Input:
        ------
        - mask: The mask of the object on the bounding box based on Otsu method
        - disp: The disparity map
        - bbox: The bounding box position of the detected objects
        - focal_lenght: The focal lenght of the used camera (default:1.93e-3)
        - baseline: The baseline of the setup used (default:49.867050e-3)
        - pixel_size: The size of a pixel in the cell of the used camera (default:3e-6)

        Output:
        -------
        - distance: The distance in meter of the object in the center of the box
        based of the histogram of the disparity value in 9/10 of the size of the
        bounding box.
        """
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
