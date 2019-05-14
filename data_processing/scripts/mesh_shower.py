#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:27:32 2019

@author: Latour Rodolphe
"""

import numpy as np
import matplotlib.pyplot as plt
import rospy
from data_processing.msg import PointMsg

class mesh_shower_2D:
    def __init__(self):
        rospy.init_node('mesh_shower_node')
        sub = rospy.Subscriber('/object/position/3D', PointMsg, self.process)
        rospy.spin()
        
    def process(self, msg):
        
        stamp = msg.header.stamp
        time = stamp.secs + stamp.nsecs * 1e-9
        x = msg.center.x
        y = msg.center.y
        z = msg.center.z
        plt.subplot(211)
        plt.plot(z, x, 'o')
        plt.subplot(212)
        plt.plot(y, x, 'o')
        plt.axis("equal")
        plt.draw()
        plt.pause(0.00000000001)
        
if __name__ == '__main__':
    try:
        mesh_shower_2D()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')