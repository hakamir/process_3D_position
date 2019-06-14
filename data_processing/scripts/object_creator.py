#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, sin, pi, sqrt, exp
import numpy as np
import time
from pyquaternion import Quaternion
"""
@author: Latour Rodolphe
"""
"""
This script creates a rectangular prism with five parameters : 
    - center of the mesh
    - scale
    - rotation (with quaternion)
    - class of the detected object
    - score of the detection
    
It corresponds to the existence area used in the areaMaker.py scripts. 
"""

class object_creator():
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

        
    def calculate_position(self):
    
        self.point_pos = self.quaternion.rotate(self.point_pos)
        self.point_pos = np.matrix([self.point_pos[0],self.point_pos[1],self.point_pos[2]]).T
        self.center = self.cam_pos + self.point_pos
        self.set_vertices()
        return

    
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
        if -self.scale[0] <= point[0] - self.center[0] <= self.scale[0] and -self.scale[1] <= point[1] - self.center[1] <= self.scale[1] and -self.scale[2] <= point[2] - self.center[2] <= self.scale[2]:
            return True
        else:
            return False

    
    def add_point(self, point, cam_pos, quaternion, score):
        for k in range(3):
            self.point_pos[k] = np.mean(point[k])
        self.quaternion = quaternion
        self.calculate_position()
        self.iteration += 1
        self.update_score(score)
        self.last_detection = time.time()
        return
    
    
    def sigmoid(self, x, k):
        return 1/(1 + exp(-x*k))
    
    def update_score(self, score):
        tmp = self.score + score
        try:
            self.score = self.sigmoid(tmp**self.iteration, 0.2)
        except OverflowError:
            self.iteration = 0
            print("Reset iteration to 0 due to OverflowError.")
        return self.score
    
    def get_center(self):
        return self.center
    
    def get_scale(self):
        return self.scale
    
    def get_quaternion(self):
        return self.quaternion
    
    def get_class(self):
        return self._class
    
    def get_score(self):
        return self.score
        
    def get_ID(self):
        return self.ID
    
    def get_vertice_coordinate(self, vertice):
        return self.vertices[vertice]
    
    def get_iteration(self):
        return self.iteration
    
    def get_creation_time(self):
        return self.creation_time
        
    def get_last_detection_time(self):
        return self.last_detection
    
    def set_last_detection_time(self):
        self.last_detection = time.time()
        return
    
    
