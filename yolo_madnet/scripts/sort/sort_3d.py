#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2
import random



global width,height,f,sx

width=1280
height=720
f=1.93
sx=3e-3

@jit


def iou(bb_test,bb_gt,mesured_distance,predicted_distance):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  zz1=np.maximum(mesured_distance,predicted_distance)
  zz2=np.minimum(mesured_distance+2,predicted_distance+2)
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  d=np.maximum(0., zz2 - zz1)
  whd = w * h*d
  o = whd / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])*2
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])*2 - whd)
  return(o)

def convert_bbox_to_z(bbox,distance):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and a distance and returns z in the form
    [X,Y,Z,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s=w*h
  r = w/float(h)
  X=(x-width/2)*distance*sx/float(f)
  Y=(y-height/2)*distance*sx/float(f)
  Z=distance
  return np.array([X,Y,Z,s,r]).reshape((5,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [X,Y,Z,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """

  x_centre=x[0]*f/(x[2]*sx)+width/2
  y_centre=x[1]*f/(x[2]*sx)+height/2
  w = np.sqrt(abs(x[3]*x[4]))
  h = x[3]/w
  x1=x_centre-w/2.
  y1=y_centre-h/2.
  x2=x_centre+w/2.
  y2=y_centre+h/2.
  if(score==None):
    return np.array([x1,y1,x2,y2]).reshape((1,4))
  else:
    return np.array([x1,y1,x2,y2,score]).reshape((1,5))

def convert_3d_prediction_to_img_coord(x):
    """
    takes 3d coord in the form [X,Y,Z] and return a [u,v] array
    """
    u=x[0]*f/(x[2]*sx)+width/2
    v=x[1]*f/(x[2]*sx)+height/2
    return [int(u),int(v)]

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,distance):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=9, dim_z=5)
    self.kf.F = np.array([[1,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,1,0],[0,0,0,1,0,0,0,0,1],  [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]])
    self.kf.R[3:,3:] *= 10.
    self.kf.R[2,2]*=130.
    self.kf.P[5:,5:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[-2,-2] *= 1e-2
    self.kf.Q[2,2] *= 0.01
    self.kf.Q[5:,5:] *= 0.01

    self.kf.x[:5] = convert_bbox_to_z(bbox,distance)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.class_id = int(bbox[-1])
    self.score = bbox[-2]

    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.valided=False

  def update_detection_score(self, dets):
      self.score = dets[4]
      return
  def update(self,bbox,distance):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox,distance))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[8]+self.kf.x[3])<=0):
      self.kf.x[8] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def x_prediction(self):
      self.kf.predict()
      return self.kf.x

  def is_valid(self,min_hits):
      if self.hit_streak >= min_hits:
          self.valided=True


  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,predicted_distances,mesured_distances,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    mesured_distance=mesured_distances[d]
    for t,trk in enumerate(trackers):
      predicted_distance=predicted_distances[t]
      iou_matrix[d,t] = iou(det,trk,mesured_distance,predicted_distance)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=3,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets,distances):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,class_id],[x1,y1,x2,y2,score,class_id],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    #TODO Link class name to ID returned on trackers and so on
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    pred=[]
    predicted_state=[]
    state_x=[]
    output=[]
    predicted_distances=[]
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      state=self.trackers[t].kf.x
      distance=state[2]
      predicted_distances.append(distance)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    if len(predicted_distances)>0:
        predicted_distances=np.concatenate(predicted_distances)
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,predicted_distances,distances)
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0],distances[d][0])
        trk.update_detection_score(dets[d,:][0])


    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:],distances[i])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        trk.is_valid(self.min_hits)
        d = trk.get_state()[0]
        current_state=trk.kf.x
        compa_prediction=np.array([[current_state[0]+current_state[5]],
                                   [current_state[1]+current_state[6]],
                                   [current_state[2]+current_state[7]],
                                   [current_state[3]+current_state[8]],
                                   [current_state[4]],
                                   [current_state[5]],
                                   [current_state[6]],
                                   [current_state[7]],
                                   [current_state[8]]])[:,:,0]
#        c_future_pos=trk.x_prediction()
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1],[trk.score],[trk.class_id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        elif(trk.valided):
            pred.append(np.concatenate((d,[trk.id+1],[trk.score],[trk.class_id])).reshape(1,-1))
        if (trk.valided):
            predicted_state.append(np.concatenate((compa_prediction[:,0],[trk.id+1])).reshape(1,-1))
            state_x.append(np.concatenate((current_state[:,0],[trk.id+1])).reshape(1,-1))



        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    output.append((ret,pred,predicted_state,state_x))
    return output


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--output', dest='output', default='output/calibv3_vitesse/',action='store_true')
    args = parser.parse_args()
    return args
