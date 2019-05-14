#!/usr/bin/python

import rospy
from data_processing.msg import ObjectMsg
from math import pi, tan

"""
This script is used to transform x and y pixel position and the z disparity to meters
"""

class spacialize:
    
    def __init__(self):
        
        rospy.init_node('spacialize_node')
        sub = rospy.Subscriber('/object/post', ObjectMsg, self.process)
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
        
    def kalman_filter(self):
        
        
        
    def process(self, msg):
        
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        _class = msg.obj_class
        score = msg.score
        
        fx = (self.width/2)/tan(self.HFOV/2 * pi/180)		# Focal in pixel (x)
        fy = (self.height/2)/tan(self.VFOV/2 * pi/180)	# Focal in pixel (y)
        if z != 0:
            z = (self.focal*self.baseline)/(self.pixel_size*z) # convert disparity to meters
            x = (x - self.width/2) * z / fx
            y = (y - self.height/2) * z / fy
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
        rospy.logerr('Could not start node.')
