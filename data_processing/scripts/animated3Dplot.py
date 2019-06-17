#!/usr/bin/python3

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
#import pyqtgraph as pg
import sys
import csv
#import numpy as np
import math

import rospy
from data_processing.msg import PointMsg
from nav_msgs.msg import Odometry
#from pyquaternion import Quaternion

"""
The MyGLView class is created to add specific functions to the GLViewWidget
from OpenGL, especially to print text on the viewer.
"""
class MyGLView(gl.GLViewWidget):
    def __init__(self, x=None, y=None, z=None, text=None):
        super(MyGLView, self).__init__()
        self.text = text
        self.x = x
        self.y = y
        self.z = z

    def showSelectedItem(self, x, y, box):
        """
        This method give the position of a selected item in the viewer.
        """
        pos = [x, y, 20, 20]
        item_list = self.itemsAt(pos)
        if not item_list == []:
            for item in item_list:
                # Get position of the item to set text to it
                if 'GLBoxItem' in str(type(item)):
                    mat = item.viewTransform().data()
                    posX, posY, posZ = mat[12], mat[13], mat[14]
                    return posX, posY, posZ
        else:
            return

    def setText(self, x, y, z, text):
        """
        The method set the position (x,y,z) and the content of the a text that might
        be used with the renderText inner function.
        """
        self.text = text
        self.x = x
        self.y = y
        self.z = z
        self.update()

    def paintGL(self, *args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        try:
            self.renderText(self.x, self.y, self.z, self.text)
        except TypeError:
            return
"""
The Box class define objects with specific parameters that correspond to
detected objects. It also create a mesh that can be shown on the viewer.
"""
class Box:
    def __init__(self, x, y, z, _class, score, ID):

        self.x = x
        self.y = y
        self.z = z
        self.x_old = 0
        self.y_old = 0
        self.z_old = 0
        self._class = _class
        self.score = score
        self.ID = ID
        self.drawn = False
        self.draw_box()

    def draw_box(self):
        """
        The method create a mesh of box type that is used to represent objects.
        """
        self.box = gl.GLBoxItem()
        # Define scale of the box
        self.scale = [1,1,1]
        self.box.setSize(self.scale[0], self.scale[1], self.scale[2])
        # Define color of the box according to class and score
        color = self.get_color_class(self._class, self.score)
        self.box.setColor(color)
        # Translate object to coordinate of the center
        self.box.translate(self.x - self.scale[0]/2, self.y - self.scale[1]/2, self.z - self.scale[2]/2)
        self.drawn = True
        return self.box

    def get_color_class(self, _class, score):
        """
        This method aimed to set specific colors to objects from a csv file.
        If the object isn't define within, set white.
        """
        with open('../data/color.csv') as f:
            reader = csv.reader(f)
            next(reader)
            data =[]
            for row in reader:
                data.append(row)
            for row in range(len(data)):
                if _class == data[row][0]:
                    return float(data[row][1]),float(data[row][2]),float(data[row][3]),score*255
            return 255,255,255,score*255

"""
Main class showing objects in a 3D environment with OpenGL.
"""
class Visualizer(object):
    def __init__(self):

        self.app = QtGui.QApplication(sys.argv)
        self.mouse = QtGui.QCursor()
        self.pxl = QtGui.QLabel('test')

        # Create custom GLViewWidget
        self.w = MyGLView()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('Mesh shower 3D')
        self.w.setBackgroundColor(0,20,70,0)
        self.width = 1280
        self.height = 720
        self.w.setGeometry(0, 0, self.width, self.height)
        self.w.show()

        # Create an axis in the world coordinate center
        axis = gl.GLAxisItem()
        axis.setSize(2,2,2)
        self.w.addItem(axis)

        # Create an axis representing the camera
        self.cam = gl.GLAxisItem()
        self.cam.setSize(1,1,1)
        self.w.addItem(self.cam)
        self.cam_x_old = 0
        self.cam_y_old = 0
        self.cam_z_old = 0
        self.rotX_old = 0
        self.rotY_old = 0
        self.rotZ_old = 0

        # create the background grids
        gz = gl.GLGridItem()
        gz.setSize(1000,1000)
        self.w.addItem(gz)

        # Initialize class variables
        self.lock = True
        self.boxes = []

        # init ROS tools
        rospy.init_node('mesh_3D_node')
        rospy.Subscriber('/object/position/3D', PointMsg, self.process, queue_size=10)
        rospy.Subscriber('/t265/odom/sample', Odometry, self.update_cam_position, queue_size=10)

    def process(self, msg):
        """
        Add a box with received data from ROS topic
        """
        x, y, z = msg.center.x, msg.center.y, msg.center.z

        _class = msg.obj_class
        score = msg.score
        ID = msg.ID
        self.box = Box(z, x, y, _class, score, ID)
        self.lock = False
        self.box.box.translate(z, x, y)
        return

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert quaternion to euler angles in degrees. The method return the
        Euler vector in the form: "yaw, pitch, roll". These values are float.
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(t0, t1))
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.degrees(math.asin(t2))
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(t3, t4))
        return yaw, pitch, roll

    def update_cam_position(self, msg):
        """
        Move the Intel T265 camera on the GLViewWidget to the known position received.
        """
        # Get the camera position (euler vector) and rotation (quaternion)
        self.cam_x = msg.pose.pose.position.z
        self.cam_y = msg.pose.pose.position.x
        self.cam_z = msg.pose.pose.position.y
        #cam_point = np.matrix([[self.cam_x], [self.cam_y], [self.cam_z]])

        self.cam_rx = msg.pose.pose.orientation.x
        self.cam_ry = msg.pose.pose.orientation.y
        self.cam_rz = msg.pose.pose.orientation.z
        self.cam_rw = msg.pose.pose.orientation.w
        #quaternion = Quaternion(self.cam_rw, self.cam_rx, self.cam_ry, self.cam_rz)

        # Translate to known position of the camera
        self.cam.translate(self.cam_x - self.cam_x_old, self.cam_y - self.cam_y_old, self.cam_z - self.cam_z_old)
        self.cam_x_old = self.cam_x
        self.cam_y_old = self.cam_y
        self.cam_z_old = self.cam_z

        # Rotate the camera from the given quaternion --> Doesn't work. It needs to be change entirely
        """
        rotX, rotY, rotZ = self.quaternion_to_euler(self.cam_rx, self.cam_ry, self.cam_rz, self.cam_rw)
        self.cam.rotate(rotX - self.rotX_old,0,0,1)
        self.cam.rotate(rotY - self.rotY_old,1,0,0)
        self.cam.rotate(rotZ - self.rotZ_old,0,1,0)
        self.rotX_old = rotX
        self.rotY_old = rotY
        self.rotZ_old = rotZ
        """

    def update(self):
        """
        This method update the scene from the ROS messages received and is able
        to move the various objects and show information about it with the cursor.
        """
        IDs = []
        # We want here to show some information about the selected object
        for box in self.boxes:
            IDs.append(box.ID)
            cursor_x = self.mouse.pos().x() - self.w.pos().x()
            cursor_y = self.mouse.pos().y() - self.w.pos().y()
            # If the cursor is on a box, then show info, else, show nothing
            try:
                posX, posY, posZ = self.w.showSelectedItem(cursor_x,cursor_y, box)
                inX = box.x - box.scale[0] <= posX <= box.x + box.scale[0]
                inY = box.y - box.scale[1] <= posY <= box.y + box.scale[1]
                inZ = box.z - box.scale[2] <= posZ <= box.z + box.scale[2]
                if inX and inY and inZ:
                    info = (box._class + ' ' + str(round(box.score,2)))
                    self.w.setText(posX, posY, posZ,info)
            except:
                self.w.setText(0, 0, 0,'')
        try:
            if not self.lock:
                if self.box.ID not in IDs:
                    print('New object added: {}, ID: {}'.format(self.box._class, self.box.ID))
                    self.w.addItem(self.box.draw_box())
                    self.boxes.append(self.box)
                    self.lock = True
                for box in self.boxes:
                    # If the new box has the same ID and class than one of the listed boxes, then correct position
                    if self.box.ID == box.ID and self.box._class == box._class:
                        box.x = self.box.x
                        box.y = self.box.y
                        box.z = self.box.z
                        box.box.translate(box.x - box.x_old, box.y - box.y_old, - (box.z - box.z_old))
                        box.x_old = box.x
                        box.y_old = box.y
                        box.z_old = box.z
                        print('Object of type {} with ID {} is moving ({}, {}, {})'.format(self.box._class, self.box.ID, self.box.x,self.box.y, self.box.z))
                        self.lock = True
        except AttributeError: # If no object has been created
            return False

    def animation(self):
        # Animate the scene each t milliseconds passing through the update method.
        t = 50
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(t)
        self.start()

    def start(self):
        # Execute the instance
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()
v.animation()
