#!/usr/bin/env python3

from __future__ import print_function
import rospy
from data_processing.msg import ObjectMsg
from data_processing.msg import PostObjectMsg
from math import pi, tan
#import sort_3d
import numpy as np

"""
This script is used to transform x and y pixel positions and the z disparity to meters
"""

class spacialize:

    def __init__(self):

        rospy.init_node('spacialize_node')
        sub = rospy.Subscriber('/object/post', PostObjectMsg, self.process_old, queue_size=1)
        self.pub = rospy.Publisher('/object', ObjectMsg, queue_size = 10)
        self.obj = ObjectMsg()

        # Parameters of the Intel D435
        self.HFOV = 91.2
        self.VFOV = 65.5
        self.DFOV = 100.6
        self.baseline = 50.0e-3
        self.focal = 1.93e-3
        self.pixel_size = 3e-6
        self.height = 720
        self.width = 1280
        rospy.spin()


    def process(self, msg):
        """
        This is a advanced approach to convert (u,v) pixel position to meters.
        It used the SORT program modified to handle 3D data and performs
        prediction from the Kalman filter.
        """
        dets = np.array([[msg.x1, msg.y1, msg.x2, msg.y2]])
        disparity = [msg.disparity]
        _class = msg.obj_class
        score = msg.score
        if disparity != 0:
            mot_tracker = sort_3d.Sort()
            trackers,predictions,trackers_space,predictions_space = mot_tracker.update(dets,disparity)
            now = rospy.get_rostime()
            self.obj.header.stamp.secs = now.secs
            self.obj.header.stamp.nsecs = now.nsecs
            self.obj.position.x = trackers_space[0]
            self.obj.position.y = trackers_space[1]
            self.obj.position.z = trackers_space[2]
            self.obj.obj_class = _class
            self.obj.score = score
            self.pub.publish(self.obj)
            return True
        else:
            print("point is at an infinite distance. Process avoided.")
            return False



    def process_old(self, msg):
        """
        This is a basic approach to convert (u,v) pixel position to meters.
        """
        u = msg.position.x
        v = msg.position.y
        disparity = msg.position.z
        scale = msg.scale
        ratio = msg.ratio
        _class = msg.obj_class
        score = msg.score

        fx = (self.width/2)/tan(self.HFOV/2 * pi/180)		# Focal in pixel (x)
        fy = (self.height/2)/tan(self.VFOV/2 * pi/180)	# Focal in pixel (y)

        if disparity != 0:

            #z = (self.focal*self.baseline)/(self.pixel_size*disparity) # convert disparity to meters
            z = disparity
            x = (u - self.width/2) * z / fx
            y = (v - self.height/2) * z / fy

            now = rospy.get_rostime()
            self.obj.header.stamp.secs = now.secs
            self.obj.header.stamp.nsecs = now.nsecs
            self.obj.position.x = x
            self.obj.position.y = y
            self.obj.position.z = z
            self.obj.obj_class = _class
            self.obj.score = score
            self.pub.publish(self.obj)
            return True
        else:
            print("point is at an infinite distance. Process avoided.")
            return False


if __name__ == '__main__':
    try:
        spacialize()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start spacialize node.')
