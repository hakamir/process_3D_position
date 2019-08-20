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
        #self.set_vertices()
        return

    """
    DEPRECATED
    This function creates the eight vertices of the box corresponding to the object.
    """
    def set_vertices(self):

        # Set vertices
        self.v1 = self.quaternion.rotate(np.matrix([[self.scale[0]],[self.scale[1]],[self.scale[2]]]))
        self.v1 = self.center + np.matrix([self.v1[0],self.v1[1],self.v1[2]])

        self.v2 = self.quaternion.rotate(np.matrix([[-self.scale[0]],[self.scale[1]],[self.scale[2]]]))
        self.v2 = self.center + np.matrix([self.v2[0],self.v2[1],self.v2[2]])

        self.v3 = self.quaternion.rotate(np.matrix([[-self.scale[0]],[-self.scale[1]],[self.scale[2]]]))
        self.v3 = self.center + np.matrix([self.v3[0],self.v3[1],self.v3[2]])

        self.v4 = self.quaternion.rotate(np.matrix([[self.scale[0]],[-self.scale[1]],[self.scale[2]]]))
        self.v4 = self.center + np.matrix([self.v4[0],self.v4[1],self.v4[2]])

        self.v5 = self.quaternion.rotate(np.matrix([[self.scale[0]],[self.scale[1]],[-self.scale[2]]]))
        self.v5 = self.center + np.matrix([self.v5[0],self.v5[1],self.v5[2]])

        self.v6 = self.quaternion.rotate(np.matrix([[-self.scale[0]],[self.scale[1]],[-self.scale[2]]]))
        self.v6 = self.center + np.matrix([self.v6[0],self.v6[1],self.v6[2]])

        self.v7 = self.quaternion.rotate(np.matrix([[-self.scale[0]],[-self.scale[1]],[-self.scale[2]]]))
        self.v7 = self.center + np.matrix([self.v7[0],self.v7[1],self.v7[2]])

        self.v8 = self.quaternion.rotate(np.matrix([[self.scale[0]],[-self.scale[1]],[-self.scale[2]]]))
        self.v8 = self.center + np.matrix([self.v8[0],self.v8[1],self.v8[2]])

        self.vertices = np.array([self.v1, self.v2, self.v3, self.v4,
                                  self.v5, self.v6, self.v7, self.v8])
        return


    def is_inside(self, point):
        """
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


    def add_point(self, point, cam_pos, quaternion, score, scale):
        """
        Description:
        ============
        The following function set the new position of the object based of a
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
