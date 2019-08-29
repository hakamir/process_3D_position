#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""

import rospy
import message_filters
from data_processing.msg import ObjectMsg
from data_processing.msg import ObjectsMsg
from yolo_madnet.msg import PointsMsg
from nav_msgs.msg import Odometry

from object_creator import object_creator
import numpy as np
from colorama import Fore, Style
from pyquaternion import Quaternion
import time



global DECAYING_TIME
DECAYING_TIME = 5

class areaMaker:
    """
    Description:
    ============
    This script is built to filter the points taken from /object/detected topic. It
    creates areas that have in attributes a name depending on the class of the
    detected object, a location, a rotation a scale, a creation time, a deadline,
    and an existence probability.

    The goal of an existence area is to classify added points to avoid overfeedings
    of point objects. It means that with recurrence of detection, the program gives
    the ability to create a coercition from previous added point in a specific
    area. Also, from the score of the detection and the recurrence, we can set an
    existence probability that can be used to avoid error of detections.
    """
    def __init__(self):

        print('Initializing node...')
        rospy.init_node('areamaker_node')
        subObj = message_filters.Subscriber('/object/detected', PointsMsg)
        subcampos = message_filters.Subscriber('/t265/odom/sample', Odometry)
        ats = message_filters.ApproximateTimeSynchronizer([subObj, subcampos], queue_size=2, slop=0.01)
        ats.registerCallback(self.process)
        self.obj_list = []
        self.pub = rospy.Publisher('/object/position/3D', ObjectsMsg, queue_size=1)
        self.time =time.time()
        rospy.spin()


    def transpose_to_global(self, point, cam_point, quaternion):
        """
        Description:
        ============
        This function is used to transpose the position of the point into global
        referential based on the tracking camera position.

        Input:
        ------
        - point: The position of the point (x,y,z) based on the camera referential
        - cam_point: The position of the tracking camera T265
        - quatertion: The rotation of the tracking camera T265

        Output:
        -------
        - point: The position of the input point set in global referential
        """
        # Perform rotation based on T265 quaternion
        point = quaternion.rotate(point)
        # Convert to numpy matrix format
        point = np.matrix([[point[0],
                            point[1],
                            point[2],
                            1]])
        cam_point = np.matrix([[1,0,0,cam_point[0][0]],
                    [0,1,0,cam_point[1][0]],
                    [0,0,1,cam_point[2][0]],
                    [0,0,0,1]])
        point = cam_point * point.T
        return point[:3]

    def process(self, pointsMsg, cameraPosMsg):
        """
        Description:
        ============
        Main function of the class.

        Input:
        ------
        - pointMsg: A list of all detected object. Each element is built
        that way:
            * Vector3 position: (x, y, z) position based of camera referential

            * Vector3 scale: 3D box scale (x, y, z) based on the bounding box.

            * string obj_class: The class of the object

            * float32 score: the detection score of the object

        - cameraPosMsg: All the data provided by the Realsense T265 camera.
        The used data here will be position (x,y,z) in meter and the orientation
        (w, i, j, k) given in quaternion.

        Output:
        -------
        - Objects: A list of object containing specific data.
            * Vector3 center:
             -- x: the x position in space (relative to the global position)
             -- y: the y position in space (relative to the global position)
             -- z: the z position in space (relative to the global position)

            * Quaternion rotation: The rotation of the object

            * Vector3 scale
             -- scale_x: the scale in x of the box based on the bounding box
             -- scale_y: the scale in y of the box based on the bounding box
             -- scale_z: the scale in z of the box based on the bounding box

            * float64 creation_time: the creation time of the object

            * float64 last_detection_time: the last detection time of the object

            * string obj_class: The class of the object

            * float32 score: the detection score of the object

            * int32 ID: The unique ID of the object
        """
        print("\n__________________________________\n")
        start = time.time()

        objects = ObjectsMsg()
        redondance = 0

        for pt in pointsMsg.point:

            # Get the point position, class and score
            # Convert point referential to camera referential
            x = pt.position.z
            y = pt.position.x
            z = pt.position.y
            print(z)
            if z > 10:
                continue
            local_point = np.matrix([[x],[y],[z]])
            _class = pt.obj_class
            score = pt.score
            id = pt.id
            scale = (pt.scale.z, pt.scale.x, pt.scale.y)


            # Get the camera position (euler vector) and rotation (quaternion)
            cam_x = cameraPosMsg.pose.pose.position.x
            cam_y = cameraPosMsg.pose.pose.position.y
            cam_z = cameraPosMsg.pose.pose.position.z
            cam_point = np.matrix([[cam_x], [cam_y], [cam_z]])

            # If tracking camera is looking ahead
            # cam_rx = cameraPosMsg.pose.pose.orientation.x
            # cam_ry = cameraPosMsg.pose.pose.orientation.y
            # cam_rz = cameraPosMsg.pose.pose.orientation.z
            # cam_rw = cameraPosMsg.pose.pose.orientation.w

            # Else if tracking camera is looking behind
            cam_rx =   cameraPosMsg.pose.pose.orientation.z
            cam_ry =   cameraPosMsg.pose.pose.orientation.w
            cam_rz = - cameraPosMsg.pose.pose.orientation.x
            cam_rw =   cameraPosMsg.pose.pose.orientation.y
            quaternion = Quaternion(cam_rw, cam_rx, cam_ry, cam_rz)


            # Print the point position in the camera and global referentials
            print(Fore.BLUE + "\nPoint position: ")
            print("#-----------REFENTIALS-----------#" + Style.RESET_ALL)
            print("Camera referential: \n{}".format(local_point))

            global_point = self.transpose_to_global(local_point, cam_point, quaternion)
            print("Global referential: \n{}".format(global_point))
            print(Fore.BLUE + "#-----------ALL OBJECTS-----------#")


            # If no object exists, we create a new mesh at the given position of the added point
            if len(self.obj_list) == 0:
                self.obj_list.append(object_creator(global_point, cam_point, scale, quaternion, _class, score, id))


            # Run through each existence area to check if the added point is inside and do processing
            for item in self.obj_list:
                # If the class of the item doesn't match with detection, skip

                if not _class == item.get_class():
                    print(_class, item.get_class())
                    lock_obj_creator = False
                    continue

                else:
                    print(Fore.BLUE + "#-----------Item {}-----------#".format(self.obj_list.index(item)) + Style.RESET_ALL)
                    print("Object center: \n{}".format(item.get_center()))
                    print("camera position: \n{}".format((cam_x,cam_y,cam_z)))
                    print("Object scale: \n{}".format(item.get_scale()))
                    print("Object class: \n{}".format(item.get_class()))
                    print("Object score: \n{}".format(round(item.get_score(),2)))
                    print("Object ID: \n{}".format(item.get_ID()))
                    iou = item.iou_3D(scale, global_point)
                    if iou >= 0.1: # and item.is_inside(global_point):
                        print(Fore.GREEN + "IoU: \n{}".format(iou) + Style.RESET_ALL)


                        # Perform calibration
                        item.calibrate(global_point, cam_point, quaternion, score, scale)
                        print("Iteration: {}".format(item.get_iteration()))

                        # Say that no object must be created
                        lock_obj_creator = True

                        # get all data Publish the position of the object
                        msg_center = item.get_center()
                        msg_scale = item.get_scale()
                        msg_rotation = item.get_quaternion()
                        msg_class = item.get_class()
                        msg_score = item.get_score()
                        msg_ID = item.get_ID()
                        msg_creation_time = item.get_creation_time()
                        msg_last_detection_time = item.get_last_detection_time()
                        now = rospy.get_rostime()
                        object = ObjectMsg()
                        object.header.stamp.secs = now.secs
                        object.header.stamp.nsecs = now.nsecs
                        object.center.x = msg_center[0]
                        object.center.y = msg_center[1]
                        object.center.z = msg_center[2]
                        object.scale.x = msg_scale[0]
                        object.scale.y = msg_scale[1]
                        object.scale.z = msg_scale[2]
                        object.rotation.w = msg_rotation[0]
                        object.rotation.x = msg_rotation[1]
                        object.rotation.y = msg_rotation[2]
                        object.rotation.z = msg_rotation[3]
                        object.creation_time = msg_creation_time
                        object.last_detection_time = msg_last_detection_time
                        object.obj_class = msg_class
                        object.score = msg_score
                        object.ID = msg_ID
                        objects.object.append(object)
                        break

                    # If the box doesn't match the existence area...
                    # let the ability to create a new one
                    else:
                        print(Fore.RED + "IoU: \n{}".format(iou) + Style.RESET_ALL)
                        lock_obj_creator = False
                    print("Creation time: \n{}".format(round(item.get_creation_time() - self.time,3)))

                    try:
                        dt = time.time() - item.get_last_detection_time()
                        print("Last detection: \n{}".format(round(dt,6)))
                        # If an object haven't been detected since a specific time, remove it.
                        if dt > DECAYING_TIME:
                            self.obj_list.remove(item)


                    # Avoid synchronization error (rare normally)
                    except AttributeError:
                        print("ERROR: Cannot get last detection time.")


            # Create a new object (a limit is added for the test)
            if not lock_obj_creator:# and len(self.obj_list) < 5:
                self.obj_list.append(object_creator(global_point, cam_point, scale, quaternion, _class, score, id))
            print("Processing time: {} ms".format(round((time.time()-start)*1000,3)))

        # Print the number of detected objects in the environment
        now = rospy.get_rostime()
        objects.header.stamp.secs = now.secs
        objects.header.stamp.nsecs = now.nsecs
        self.pub.publish(objects)
        print("Detected objects: {}".format(len(self.obj_list)))

if __name__ == '__main__':
    try:
        areaMaker()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start area maker node.')
