#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Rodolphe Latour
"""
import rospy
from visualization_msgs.msg import Marker
from data_processing.msg import ObjectsMsg
import random, time

class visualizer:
    """
    Description:
    ============
    A simple ROS node to publish 3D box in rviz.

    It manages the object data by taking in account the class, the ID, the
    position, the rotation and the scale of the object.
    """
    def __init__(self):
        rospy.init_node('visualizer_node')
        rospy.Subscriber('/object/position/3D', ObjectsMsg, self.process, queue_size=10)
        self.pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
        self.classes = self.load_classes('../../yolo_madnet/scripts/adapt/adapt.names')
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        rospy.spin()

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def process(self, msg):
        """
        Description:
        ============
        The process function. It publishes markers for each object in entry.

        Input:
        ------
        - Objects: A list of object containing specific data.
            * Vector3 center: (x, y, z) position in global referential

            * Quaternion rotation: The rotation of the object

            * Vector3 scale: (x, y, z) scale of box based on the bounding box

            * float64 creation_time: the creation time of the object

            * float64 last_detection_time: the last detection time of the object

            * string obj_class: The class of the object

            * float32 score: the detection score of the object

            * int32 ID: The unique ID of the object

        Output:
        -------
        - marker: A marker published for rviz with all the data coming in input.
        """
        for object in msg.object:
            marker = Marker()
            marker.header.frame_id = "t265_pose_frame"
            #marker.header.stamp = rospy.now()
            print(object)
            marker.ns = object.obj_class
            marker.id = object.ID

            # Set marker type as cuboid
            marker.type = 1
            # Only show cuboid on rviz if high detection score. Else, remove it.
            if object.score > 0.9:
                marker.action = 0
            else:
                marker.action = 2

            # Set position and rotation of the marker
            marker.pose.position.x = object.center.z
            marker.pose.position.y = object.center.x
            marker.pose.position.z = object.center.y
            marker.pose.orientation.x = object.rotation.x
            marker.pose.orientation.y = object.rotation.y
            marker.pose.orientation.z = object.rotation.z
            marker.pose.orientation.w = object.rotation.w

            # Set marker scale
            marker.scale.x = object.scale.z
            marker.scale.y = object.scale.x
            marker.scale.z = object.scale.y

            # Set marker color (one by class)
            color_index = self.classes.index(object.obj_class)
            marker.color.r = float(self.colors[color_index][0])/255
            marker.color.g = float(self.colors[color_index][1])/255
            marker.color.b = float(self.colors[color_index][2])/255
            marker.color.a = object.score

            marker.lifetime = rospy.Duration()
            self.pub.publish(marker)

if __name__ == '__main__':
    try:
        visualizer()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualizer node.')
