import rospy
import message_filters
from data_processing.msg import ObjectMsg
from data_processing.msg import PointMsg
from data_processing.msg import CameraMsg

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

class areaMaker:
    
    def __init__(self):
        
        print('Initializing node...')
        rospy.init_node('areamaker_node')
        
        subObj = message_filters.Subscriber('/object', ObjectMsg)
        subcampos = message_filters.Subscriber('/camera/position', CameraMsg)
        
        ats = message_filters.ApproximateTimeSynchronizer([subObj, subcampos], queue_size=10, slop=0.1)
        ats.registerCallback(self.process)
        self.obj_list = []
        self.pub = rospy.Publisher('/object/position/3D', PointMsg, queue_size=10)
        self.time =time.time()
        self.msg = PointMsg()
        rospy.spin()
    
    def transpose_to_global_quaternion(self, point, cam_point, quaternion):

        point = quaternion.rotate(point)
        point = np.matrix([point[0],point[1],point[2]])
        point = cam_point + point.T
        return point
    
    def process(self, objectMsg, cameraPosMsg):
        """
        Main function of the class. Perform the job of the script and publish
        in /object/position/3D topic the whole position of every detected
        objects. It performs like a light SLAM. 
        """
        print("\n__________________________________\n")
        start = time.time()
        
        # Get the point position, class and score
        x = objectMsg.position.x
        y = objectMsg.position.y
        z = objectMsg.position.z
        point = np.matrix([[x],[y],[z]])
        _class = objectMsg.obj_class
        score = objectMsg.score
        ID = objectMsg.ID
        
        # Get the camera position (euler vector) and rotation (quaternion)
        cam_x = cameraPosMsg.linear.x
        cam_y = cameraPosMsg.linear.y
        cam_z = cameraPosMsg.linear.z
        cam_point = np.matrix([[cam_x], [cam_y], [cam_z]])
        
        cam_rx = cameraPosMsg.angular.x
        cam_ry = cameraPosMsg.angular.y
        cam_rz = cameraPosMsg.angular.z
        cam_rw = cameraPosMsg.angular.w
#        cam_rot = np.matrix([[cam_rx], [cam_ry], [cam_rz], [cam_rw]])
        
        quaternion = Quaternion(cam_rw, cam_rx, cam_ry, cam_rz)
        
        # Set an arbitrary scale of the mesh
        # MUST BE DEPENDANT OF THE CLASS OF THE OBJECT IN THE FURE
        scale = (1,1,1)
        
        redondance = 0
        
        # Print the point position in the camera and global referentials
        print(Fore.BLUE + "\nPoint position: ")
        print("#-----------REFENTIALS-----------#" + Style.RESET_ALL)
        print("Camera referential: \n{}".format(point))
#        global_point = self.transpose_to_global_euler(point, cam_point, cam_rot)
        global_point = self.transpose_to_global_quaternion(point, cam_point, quaternion)
        print("Global referential: \n{}".format(global_point))
        print(Fore.BLUE + "#-----------ALL OBJECTS-----------#")
        
        # If no object exist, we create a new mesh at the given position of the added point
        if len(self.obj_list) == 0:
            self.obj_list.append(object_creator(point, cam_point, scale, quaternion, _class, score, ID))
        
        # Run through each existence area to check if the added point is inside and do processing
        for item in self.obj_list:
            print(Fore.BLUE + "#-----------Item {}-----------#".format(self.obj_list.index(item)) + Style.RESET_ALL)
            print("Object center: \n{}".format(item.get_center()))
            print("Object class: \n{}".format(item.get_class()))
            print("Object score: \n{}".format(round(item.get_score(),2)))
            
            # If the class of the added point match with the existence area it is inside, recalibrate position by doing tracking
            if _class == item.get_class() and item.is_inside(global_point):
                print(Fore.GREEN + 'Point is inside.' + Style.RESET_ALL)
                
                # Perform calibration
                item.add_point(point, cam_point, quaternion, score)
                item.set_last_detection_time()
                print("Iteration: {}".format(item.get_iteration()))
                
                # We verify if the point is in many existence area of the same class
                redondance += 1
                print("redondance: {}".format(redondance))
                
                # We correct the duplication problem by removing an overcrafting of mesh
                if redondance > 1:
                    self.obj_list.remove(item)
                    break
                    
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
                if dt > 10:
                    self.obj_list.remove(item)
            
            # Avoid synchronization error (rare normally)
            except AttributeError:
                print("ERROR: Cannot get last detection time.")
            
            # get all data Publish the position of the object
            msg_center = item.get_center()
            msg_rotation = item.get_quaternion()
            msg_class = item.get_class()
            msg_score = item.get_score()
            msg_ID = item.get_ID()
            msg_creation_time = item.get_creation_time()
            msg_last_detection_time = item.get_last_detection_time()
            now = rospy.get_rostime()
            self.msg.header.stamp.secs = now.secs
            self.msg.header.stamp.nsecs = now.nsecs
            self.msg.center.x = msg_center[0]
            self.msg.center.y = msg_center[1]
            self.msg.center.z = msg_center[2]
            self.msg.rotation.x = msg_rotation[0]
            self.msg.rotation.y = msg_rotation[1]
            self.msg.rotation.z = msg_rotation[2]
            self.msg.creation_time = msg_creation_time
            self.msg.last_detection_time = msg_last_detection_time
            self.msg.obj_class = msg_class
            self.msg.score = msg_score
            self.msg.ID = msg_ID
            self.pub.publish(self.msg)

        # Print the number of detected objects in the environment        
        print("Detected objects: {}".format(len(self.obj_list)))
        
        # Create a new object (a limit is added for the test)
#        if not lock_obj_creator and len(self.obj_list) < 1:
        if not lock_obj_creator:
            self.obj_list.append(object_creator(point, cam_point, scale, quaternion, _class, score, ID))
        print("Processing time: {} ms".format(round((time.time()-start)*1000,3)))
        
if __name__ == '__main__':
    try:
        areaMaker()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')