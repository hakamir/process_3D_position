
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

file_dir_path="/media/antoine/Disque dur portable/Detection+Depth/Yolov3_Stereo_depth_multithread/output/Mesure_erreur_GTv3/MAD/estimation_npy/"
img_dir_path="/media/antoine/Disque dur portable/Datasets/RealSense/python/v2/frames/1/"

def get_file_number(file):
    number=os.path.splitext(file)[0]
    return number

file_list=[file_dir_path + file_name for file_name in os.listdir(file_dir_path)]
file_list=sorted(file_list,key=get_file_number)

img_list=[img_dir_path + file_name for file_name in os.listdir(img_dir_path)]
img_list=sorted(img_list,key=get_file_number)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

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
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.valided=False
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
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
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
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        elif(trk.valided):
            pred.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
        if (trk.valided):
            predicted_state.append(np.concatenate((compa_prediction[:,0],[trk.id+1])).reshape(1,-1))
            state_x.append(np.concatenate((current_state[:,0],[trk.id+1])).reshape(1,-1))



        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    output.append((ret,pred,predicted_state,state_x))
    return output
#    if(len(ret)>0):
#        if(len(pred)>0):
#            return np.concatenate(ret),np.concatenate(pred)
#        else:
#            return np.concatenate(ret),[]
#    if(len(pred)>0):
#        return [],np.concatenate(pred)
#    return np.empty((0,5)),np.empty((0,5))







def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--output', dest='output', default='output/calibv3_vitesse/',action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
  # all trai
  args = parse_args()
  if not os.path.exists(args.output):
        os.makedirs(args.output)
  if not os.path.exists(args.output+"predictions/"):
        os.makedirs(args.output+"predictions/")
#  display = args.display
  phase = 'train'
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(1000,3) #used only for display
  colours*=255


#  if not os.path.exists('output'):
#    os.makedirs('output')


  mot_tracker = Sort() #create instance of the SORT tracker
  with open(args.output+'out.txt','w') as out_file:
      history_prediction=[]
      distance_error_list=[]
      for frame in range(len(file_list)):
        original_img=cv2.imread(img_list[frame])
        data=np.load(file_list[frame])
        frame += 1 #detection and frame numbers begin at 1
        if frame==685:
            print('nique')
        if len(data)>0:
            dets=data[:,0:4]
            distances=data[:,4]
        else:
            dets=[]
            distances=[]
        total_frames += 1


        if frame==333:
            print('ok')
        start_time = time.time()
#        trackers,predictions = mot_tracker.update(dets,distances)
        trackers,predictions,predicted_state,state_x=np.array(mot_tracker.update(dets,distances)[0])
        if len(trackers)>0:
            trackers=np.concatenate(trackers)
        if len(predicted_state)>0:
            predicted_state=np.concatenate(predicted_state)
        if len(predictions)>0:
            predictions=np.concatenate(predictions)
        if len(state_x)>0:
            state_x=np.concatenate(state_x)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          x1,y1,x2,y2=int(d[0]),int(d[1]),int(d[2]),int(d[3])
#          distance=get_object_distance(self.disp,x1,y1,x2,y2,args.focal_lenght,args.baseline,args.pixel_size)
#          label=self.labels[j]+" distance= %.2fm"%distance
          label=str(d[4])
          plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(d[4])-1])

          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)


        for d in predictions:
          x1,y1,x2,y2=int(d[0]),int(d[1]),int(d[2]),int(d[3])
          label=str(d[4])+' prediction'
          plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(d[4])-1])

          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

        np.save(args.output+"predictions/pred%06i.npy"%frame,predicted_state)
        for d in predicted_state:
            idx=d[9]
#            x=d[0:3]
#            pixel_coord=convert_3d_prediction_to_img_coord(x)

            x=d[0:5]
            pixels=convert_x_to_bbox(x)[0]
            x1,y1,x2,y2=int(pixels[0]),int(pixels[1]),int(pixels[2]),int(pixels[3])
            label=str(int(idx))+' prediction'
#            plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(idx)+100])
            pixel_coord=[int((x2+x1)/2),int((y2+y1)/2)]
            if len(trackers)>0:
                row=np.where(trackers[:,4]==idx)
                if(len(row[0]))>0:
                    x1,y1,x2,y2=int(trackers[row[0][0],0]),int(trackers[row[0][0],1]),int(trackers[row[0][0],2]),int(trackers[row[0][0],3])

                    original_position=[int((x2+x1)/2.),int((y2+y1)/2.)]
                    cv2.arrowedLine(original_img,(original_position[0],
                                                  original_position[1]),(pixel_coord[0],pixel_coord[1]),
                                                  colours[int(idx)-1,:],thickness = 2)
                    vitesses=[d[5]*30,d[6]*30,d[7]*30]
                    vitesse_globale=np.sqrt(vitesses[0]**2+vitesses[1]**2+vitesses[2]**2)
#                    label="vx:%.3f m/s   vy:%.3f m/s   vz:%.3f m/s   v:%.3f m/s"%(vitesses[0],vitesses[1],vitesses[2],vitesse_globale)
                    label="v:%.3f m/s"%vitesse_globale
                    plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(idx)-1])
            if len(predictions)>0:
                row=np.where(predictions[:,4]==idx)
                if(len(row[0]))>0:
                    x1,y1,x2,y2=int(predictions[row[0][0],0]),int(predictions[row[0][0],1]),int(predictions[row[0][0],2]),int(predictions[row[0][0],3])

                    original_position=[int((x2+x1)/2.),int((y2+y1)/2.)]
                    cv2.arrowedLine(original_img,(original_position[0],
                                                  original_position[1]),(pixel_coord[0],
                                                                   pixel_coord[1]),colours[int(idx)-1,:],thickness = 2)

        history_prediction.append(predicted_state)
        distance_error=[]
        if len(history_prediction[frame-2])>0:
            if len(state_x)>0:
                for d in history_prediction[frame-2]:
                    row=np.where(state_x[:,9]==d[9])[0]
                    if len(trackers)>0:
                        if len(np.where(trackers[:,4]==d[9])[0])>0:
                            if len(row)>0:
                                distance_error.append(np.sqrt((state_x[row[0],0]-d[0])**2+(state_x[row[0],1]-d[1])**2+(state_x[row[0],2]-d[2])**2))
        distance_error_list.append(distance_error)
        cv2.imwrite(args.output+'out%06i.png'%frame,original_img)



  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  distance_error_list=np.concatenate(distance_error_list)
  mean_error=sum(distance_error_list)/len(distance_error_list)
