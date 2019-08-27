#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Latour Rodolphe
"""

from math import cos, sin, pi, sqrt, exp
import numpy as np
import time
from pyquaternion import Quaternion


class object_creator():
    """
    Description:
    ============
    This script creates a rectangular prism with five parameters :
        - center of the mesh
        - scale
        - rotation (with quaternion)
        - class of the detected object
        - score of the detection

    It corresponds to the existence area used in the areaMaker.py scripts.
    """
    def __init__(self, point_pos, cam_pos, scale, quaternion, _class, score, ID):

        self.point_pos = point_pos
        self.cam_pos = cam_pos
        self.scale = scale
        self.quaternion = quaternion
        self._class = _class
        self.score = score
        self.ID = ID
        self.iteration = 1
        self.creation_time = time.time()
        self.last_detection = self.creation_time
        self.calculate_position()

    """
    This function is used to calculate the position of the object.
    It refers the point position after the quaternion rotation and set the
    center of the box.
    """
    def calculate_position(self):

        self.point_pos = self.quaternion.rotate(self.point_pos)
        self.point_pos = np.matrix([self.point_pos[0],self.point_pos[1],self.point_pos[2]]).T
        self.center = self.cam_pos + self.point_pos
        return

    def is_inside(self, point):
        """

        !!! DEPRECATED !!!
        The iou_3D() function is used to check matching between boxes.

        Description:
        ============
        This method is used to indicate if a given point is inside the box
        corresponding to the objet.

        Input:
        ------
        - point: a point (list of x,y,z position)

        Output:
        -------
        - True: the input point is inside the object scale
        OR False: the input point is outside the object scale
        """
        inX = -self.scale[0] <= point[0] - self.center[0] <= self.scale[0]
        inY = -self.scale[1] <= point[1] - self.center[1] <= self.scale[1]
        inZ = -self.scale[2] <= point[2] - self.center[2] <= self.scale[2]
        if inX and inY and inZ:
            return True
        else:
            return False

    def iou_3D(self, scale, point):
        """
        Description:
        ============
        This method tries to catch matching between the input scale and the
        scale of the object. It is used to indicate if the provided box can
        correspond to this object.

        Input:
        ------
        - scale: a scale of a new detected object.
        - point: the point in space of the new detected object. It must be in
        global environment.

        Output:
        -------
        - IoU: The Intersection of Union is returned (value between 0 and 1). It
        corresponds to the IoU between the object box and the entry box (with
        scale and center point).
        """
        # Calculate the area of both boxes
        area = self.scale[0] * self.scale[1] * self.scale[2]
        new_area = scale[0] * scale[1] * scale[2]
        ratio = area/new_area
        # Retrieve the initial (x1,x2,y1,y2,z1,z2) positions of the new box
        x1 = point[0] - scale[0]/2
        x2 = point[0] + scale[0]/2
        y1 = point[1] - scale[1]/2
        y2 = point[1] + scale[1]/2
        z1 = point[2] - scale[2]/2
        z2 = point[2] + scale[2]/2

        # Retrieve the initial (x1,x2,y1,y2,z1,z2) positions of the actual box
        x1p = self.center[0] - self.scale[0]/2
        x2p = self.center[0] + self.scale[0]/2
        y1p = self.center[1] - self.scale[1]/2
        y2p = self.center[1] + self.scale[1]/2
        z1p = self.center[2] - self.scale[2]/2
        z2p = self.center[2] + self.scale[2]/2


        # calculate the area overlap with the previous data
        overlap = max((min(x2,x2p)-max(x1,x1p)),0)*max((min(y2,y2p)-max(y1,y1p)),0)*max((min(z2,z2p)-max(z1,z1p)),0)
        # calculate the IoU with the previous calculated area
        IoU = overlap / (area + new_area - overlap)
        if type(IoU) == np.matrix:
            return IoU.item(0)

    def add_point(self, point, cam_pos, quaternion, score, scale):
        """
        Description:
        ============
        The following function set the new position of the object based on a
        given point. It also updates the existence probability score, the scale
        of the object, increase the detection iteration by one, set the rotation
        of the object on the tracking camera quaternion value and set the last
        detection time.

        Input:
        ------
        - point: the new position of the object
        - cam_pos: the position of the tracking camera T265
        - quaternion: the rotation of the tracking camera T265
        - score: the detection score of the object
        - scale: the input scale of the object (x, y, z)
        """
        for k in range(3):
            #self.point_pos[k] = np.mean(point[k])
            self.point_pos[k] = point[k]
        self.quaternion = quaternion
        self.scale = scale
        self.calculate_position()
        self.iteration += 1
        self.update_score(score)
        self.last_detection = time.time()
        return


    def sigmoid(self, x, k):
        """
        Description:
        ============
        A simple sigmoid function. It calculates the sum of the score to the
        power of the number of iteration.

        The more the iteration will be and the higher the score is, the more
        possible the object exists.

        Input:
        ------
        - x: the sum of the score
        - k: the iteration number
        """
        return 1/(1 + exp(-x*k))


    def update_score(self, score):
        """
        Description:
        ============
        This method updates the existence probability of the object based on the
        detection score as input. It takes in account every score given by iteration
        and calculate the sigmoid (with factor 0.2) of the sum of scores to the power
        of the number of iteration of detection.

        Input:
        ------
        - score: The score of the detection

        Output:
        -------
        - score: The probability existence of the object
        """
        tmp = self.score + score
        try:
            self.score = self.sigmoid(tmp**self.iteration, 0.2)
        except OverflowError:
            self.iteration = 0
            print("Reset iteration to 0 due to OverflowError.")
        return self.score


    def get_center(self):
        """
        Return the center of the box
        """
        return self.center


    def get_scale(self):
        """
        Return the scale of the box
        """
        return self.scale


    def get_quaternion(self):
        """
        Return the rotation of the box
        """
        return self.quaternion


    def get_class(self):
        """
        Return the class of the object
        """
        return self._class


    def get_score(self):
        """
        Return the existence probability based on the recurrent detection
        """
        return self.score


    def get_ID(self):
        """
        Return the ID of the object
        """
        return self.ID


    def get_vertice_coordinate(self, vertice):
        """
        !!!DEPRECATED!!!
        Return the vertices coordinates.
        """
        return self.vertices[vertice]


    def get_iteration(self):
        """
        Return the number of times the object has been detected
        """
        return self.iteration


    def get_creation_time(self):
        """
        Return the time of the first detection
        """
        return self.creation_time


    def get_last_detection_time(self):
        """
        Return the time of the last detection
        """
        return self.last_detection


    def set_last_detection_time(self):
        """
        Set the time of the last detection as now
        """
        self.last_detection = time.time()
        return
