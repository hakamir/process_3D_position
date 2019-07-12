#!/usr/bin/python3

import rospy
import message_filters
from data_processing.msg import ObjectMsg
from data_processing.msg import ObjectsMsg
from data_processing.msg import PointsMsg
from nav_msgs.msg import Odometry

from object_creator import object_creator
import numpy as np
from colorama import Fore, Style
from pyquaternion import Quaternion
import time
"""
@author: Latour Rodolphe
"""
"""
This script is build to filter the points taken from /object/position/meters
topic. It create areas that have in attributes a name depending the class of
the detected object, a location, a rotation a scale, a creation time,
a deadline, and an existence probability.

The goal of an existence area is to classify added points to avoid overfeedings
of point objects. It means that with recurrence of detection, the program gives
the ability to create a coercition from previous added point in a specific
area. Also, from the score of the detection and the recurrence, we can set an
existence probability that can be used to avoid error of detections.

Program in construction...
"""

global DECAYING_TIME
DECAYING_TIME = 120

class areaMaker:

    def __init__(self):

        print('Initializing node...')
        rospy.init_node('areamaker_node')
        self.IDs = []
        subObj = message_filters.Subscriber('/object/detected', PointsMsg)
        subcampos = message_filters.Subscriber('/t265/odom/sample', Odometry)
        ats = message_filters.ApproximateTimeSynchronizer([subObj, subcampos], queue_size=2, slop=0.01)
        ats.registerCallback(self.process)
        self.obj_list = []
        self.pub = rospy.Publisher('/object/position/3D', ObjectsMsg, queue_size=1)
        self.time =time.time()
        rospy.spin()

    def transpose_to_global_quaternion(self, point, cam_point, quaternion):

        point = quaternion.rotate(point)
        point = np.matrix([point[0],point[1],point[2]])
        point = cam_point + point.T
        return point

    def process(self, pointsMsg, cameraPosMsg):
        """
        Main function of the class. Perform the job of the script and publish
        in /object/position/3D topic the whole position of every detected
        objects. It performs like a light SLAM.
        """
        print("\n__________________________________\n")
        start = time.time()

        objects = ObjectsMsg()
        redondance = 0

        for pt in pointsMsg.point:
            # Get the point position, class and score
            x = pt.position.x
            y = pt.position.y
            z = pt.position.z
            point = np.matrix([[x],[y],[z]])
            _class = pt.obj_class
            score = pt.score
            scale = (pt.scale.x, pt.scale.y, pt.scale.z)

            # Get the camera position (euler vector) and rotation (quaternion)
            cam_x = cameraPosMsg.pose.pose.position.x
            cam_y = cameraPosMsg.pose.pose.position.y
            cam_z = cameraPosMsg.pose.pose.position.z
            cam_point = np.matrix([[cam_x], [cam_y], [cam_z]])

            cam_rx = cameraPosMsg.pose.pose.orientation.x
            cam_ry = cameraPosMsg.pose.pose.orientation.y
            cam_rz = cameraPosMsg.pose.pose.orientation.z
            cam_rw = cameraPosMsg.pose.pose.orientation.w
            quaternion = Quaternion(cam_rw, cam_rx, cam_ry, cam_rz)

            # The T265 tracking camera was set looking backward.
            # Then, the X and Z axis are inverted.
            cam_x = - cam_x
            cam_z = - cam_z
            # The rotation might be taken in consideration
            #TODO
            #quaternion.rotate()

            #scale = (1,1,1)

            # Print the point position in the camera and global referentials
            print(Fore.BLUE + "\nPoint position: ")
            print("#-----------REFENTIALS-----------#" + Style.RESET_ALL)
            print("Camera referential: \n{}".format(point))
            global_point = self.transpose_to_global_quaternion(point, cam_point, quaternion)
            print("Global referential: \n{}".format(global_point))
            print(Fore.BLUE + "#-----------ALL OBJECTS-----------#")

            # If no object exists, we create a new mesh at the given position of the added point
            if len(self.obj_list) == 0:
                self.obj_list.append(object_creator(point, cam_point, scale, quaternion, _class, score, len(self.obj_list)))

            # Run through each existence area to check if the added point is inside and do processing
            for item in self.obj_list:
                print(Fore.BLUE + "#-----------Item {}-----------#".format(self.obj_list.index(item)) + Style.RESET_ALL)
                print("Object center: \n{}".format(item.get_center()))
                print("Object scale: \n{}".format(item.get_scale()))
                print("Object class: \n{}".format(item.get_class()))
                print("Object score: \n{}".format(round(item.get_score(),2)))
                print("Object ID: \n{}".format(item.get_ID()))

                # If the class of the added point match with the existence area it is inside, recalibrate position by doing tracking
                if _class == item.get_class() and item.is_inside(global_point):
                    print(Fore.GREEN + 'Point is inside.' + Style.RESET_ALL)

                    # Perform calibration
                    item.add_point(point, cam_point, quaternion, score, scale)
                    print("Iteration: {}".format(item.get_iteration()))

                    # We verify if the point is in many existence area of the same class
                    redondance += 1

                    # We correct the duplication problem by removing an overcrafting of mesh
                    if redondance > 1:
                        self.obj_list.remove(item)
                        break
                    print("redondance: {}".format(redondance))

                    # Say that no object must be created
                    lock_obj_creator = True

                # If the point is not in an existence area, let the ability to create a new one
                else:
                    print(Fore.RED + "Point is out!")
                    print(Style.RESET_ALL)
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

            # Print the number of detected objects in the environment
            now = rospy.get_rostime()
            objects.header.stamp.secs = now.secs
            objects.header.stamp.nsecs = now.nsecs
            self.pub.publish(objects)
            print("Detected objects: {}".format(len(self.obj_list)))

            # Create a new object (a limit is added for the test)
            if not lock_obj_creator and len(self.obj_list) < 50:
                self.obj_list.append(object_creator(point, cam_point, scale, quaternion, _class, score, len(self.obj_list)))
            print("Processing time: {} ms".format(round((time.time()-start)*1000,3)))

if __name__ == '__main__':
    try:
        areaMaker()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start area maker node.')
