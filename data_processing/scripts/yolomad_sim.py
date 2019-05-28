#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""
import rospy
from data_processing.msg import BboxMsg
import os
import numpy as np

file_dir_path="/home/amphani/catkin_ws/src/data_processing/data/moving/npy/"
img_dir_path="/home/amphani/catkin_ws/src/data_processing/data/moving/images/"

def get_file_number(file):
    number=os.path.splitext(file)[0]
    return number

file_list=[file_dir_path + file_name for file_name in os.listdir(file_dir_path)]
file_list=sorted(file_list,key=get_file_number)

img_list=[img_dir_path + file_name for file_name in os.listdir(img_dir_path)]
img_list=sorted(img_list,key=get_file_number)

def yolomad_sim():
    
    rospy.init_node('yolomad_sim_node')
    pub = rospy.Publisher('/object/post', BboxMsg, queue_size = 10)
    msg = BboxMsg()
    rate = rospy.Rate(5)

    
    while not rospy.is_shutdown():

        for frame in range(len(file_list)):
            data=np.load(file_list[frame])
            if len(data) != 0:
                for obj in range(len(data)):
                    now = rospy.get_rostime()
                    msg.header.stamp.secs = now.secs
                    msg.header.stamp.nsecs = now.nsecs
                    msg.x1 = data[obj,0]
                    msg.x2 = data[obj,1]
                    msg.y1 = data[obj,2]
                    msg.y2 = data[obj,3]
                    msg.disparity = data[obj,4]
                    msg.obj_class = str(data[obj,5][0:-4])
                    msg.score = float(data[obj,5][-4:])
                    pub.publish(msg)
                    rate.sleep()

if __name__ == '__main__':
    try:
        yolomad_sim()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')