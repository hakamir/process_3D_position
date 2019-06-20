
"""
Created on Mon Jun 17 11:35:01 2019

@author: antoine
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter

global width,height,f,sx,baseline
baseline=49.867050
width=1280
height=720
f=1.93
sx=3e-3


def get_file_number(file):
    number=os.path.splitext(file)[0]
    return number

@jit
def iou(bb_test,bb_gt,mesured_distance,predicted_distance):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2] and with the distance
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  zz1=np.maximum(mesured_distance,predicted_distance)
  if zz1<6:
      zz2=np.minimum(mesured_distance+(2),predicted_distance+(2))
      d=np.maximum(0., zz2 - zz1)
      w = np.maximum(0., xx2 - xx1)
      h = np.maximum(0., yy2 - yy1)
      whd = w * h*d
  
      o = whd / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])*2
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])*2 - whd)
  else:
      w = np.maximum(0., xx2 - xx1)
      h = np.maximum(0., yy2 - yy1)
      wh = w * h
      o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def get_list(file_dir_path,img_dir_path,disp_path):
    """
    Get the list of files where the detection are stored, the input images and the disparities
    """
    
    file_list=[file_dir_path + file_name for file_name in os.listdir(file_dir_path)]
    file_list=sorted(file_list,key=get_file_number)
    
    img_list=[img_dir_path + file_name for file_name in os.listdir(img_dir_path)]
    img_list=sorted(img_list,key=get_file_number)
    
    disp_list=[disp_path + file_name for file_name in os.listdir(disp_path)]
    disp_list=sorted(disp_list,key=get_file_number)
    
    return file_list,img_list,disp_list

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


def convert_im_coord_to_3d(coord,distance):
    X=(coord[0]-width/2)*distance*sx/float(f)
    Y=(coord[1]-height/2)*distance*sx/float(f)
    Z=distance
    return np.array([X,Y,Z]).reshape((3,1))

def get_object_distance(disp,x1,y1,x2,y2,focal_lenght,baseline,pixel_size):
    hist_with_indx=np.zeros((65536,2))
    hist_with_indx[:,0]=np.arange(65536)
    x1_prim=x1+9/10*(x2-x1)/2
    y1_prim=y1+9/10*(y2-y1)/2
    x2_prim=x2-9/10*(x2-x1)/2
    y2_prim=y2-9/10*(y2-y1)/2
#    histr = cv2.calcHist([disp[y1:y2,x1:x2]],[0],None,[65536],[0,65535])
    histr = cv2.calcHist([disp[int(y1_prim):int(y2_prim),int(x1_prim):int(x2_prim)]],[0],None,[65536],[0,65535])
    distance_list=[]
    hist_with_indx[:,1]=histr[:,0]
    indx_to_remove=np.argmax(hist_with_indx[:,1])
    distance=int(hist_with_indx[indx_to_remove,0])
    max_hist=hist_with_indx[indx_to_remove,1]
    hist_with_indx=np.delete(hist_with_indx,(indx_to_remove),axis=0)
    max_score=max_hist*indx_to_remove**2
    if distance==0:
        distance=0 #Pour KITTI
    else:
        distance=distance*0.001
    distance_list.append(distance)
    for i in range(3):
        indx_to_remove=np.argmax(hist_with_indx[:,1])
        score=hist_with_indx[indx_to_remove,1]*hist_with_indx[indx_to_remove,0]**2
        if score>max_score:
            max_score=score
        
        distance=int(hist_with_indx[indx_to_remove,0])
        ecart=abs(score-max_score)
        quadra=ecart/max_score
        if quadra<0.1:
            if distance==0:
                distance=0
            else:
                distance=distance*0.001
            distance_list.append(distance)
        hist_with_indx=np.delete(hist_with_indx,(indx_to_remove),axis=0)
    distance=np.sum(distance_list)/len(distance_list)
    return distance

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

def plot(state_x_list,prediction_list,detections):
    FRAME=[]
    X_state=[]
    Y_state=[]
    Z_state=[]
    X_pred=[]
    Y_pred=[]
    Z_pred=[]
    X_pred_no_Kalman=[]
    Y_pred_no_Kalman=[]
    Z_pred_no_Kalman=[]
    detection_X=[]
    detection_Y=[]
    detection_Z=[]
    vx_kalman=[]
    vz_kalman=[]
    vz_sans_kalman=[]
    vx_sans_kalman=[]
    FRAME_30=[]
    X_30=[]
    Y_30=[]
    Z_30=[]
    X_error_pred_1s=[]
    X_Err=[]
    
    for k in range(len(state_x_list)):
        if len(detections[k])>0 and len(prediction_list[k-1])>0 and len(detections[k-2])>0 and len(detections[k-1])>0 and len(state_x_list[k-1])>0:
            if detections[k][2]==detections[k-1][2]==detections[k-2][2]==int(prediction_list[k-1][0,-1])==int(state_x_list[k][0,-1]):
                
                FRAME.append(k)
                x_det_k,y_det_k,z_det_k,_,_=convert_bbox_to_z(detections[k][0],detections[k][1])
                x_det_k_1,y_det_k_1,z_det_k_1,_,_=convert_bbox_to_z(detections[k-1][0],detections[k-1][1])
                x_det_k_2,y_det_k_2,z_det_k_2,_,_=convert_bbox_to_z(detections[k-2][0],detections[k-2][1])
                X_state.append(state_x_list[k][0,0])
                Y_state.append(state_x_list[k][0,1])
                Z_state.append(state_x_list[k][0,2])
                X_pred.append(prediction_list[k-1][0,0])
                Y_pred.append(prediction_list[k-1][0,1])
                Z_pred.append(prediction_list[k-1][0,2])
                vx_kalman.append(state_x_list[k][0,5])
                vz_kalman.append(state_x_list[k][0,7])
                vx=x_det_k_1[0]-x_det_k_2[0]
                vy=y_det_k_1[0]-y_det_k_2[0]
                vz=z_det_k_1[0]-z_det_k_2[0]
                vx_sans_kalman.append(vx*30)
                vz_sans_kalman.append(vz*30)
                X_pred_no_Kalman.append(vx+x_det_k_1[0])
                Y_pred_no_Kalman.append(vy+y_det_k_1[0])
                Z_pred_no_Kalman.append(vz+z_det_k_1[0])
                detection_X.append(x_det_k[0])
                detection_Y.append(y_det_k[0])
                detection_Z.append(z_det_k[0])
            if len(state_x_list[k-30])>0 and k>29:
                FRAME_30.append(k)
                X_Err.append(abs((state_x_list[k-30][0,0]+state_x_list[k-30][0,5]+1/2*state_x_list[k-30][0,9])-state_x_list[k][0,0]))
                X_30.append(state_x_list[k-30][0,0]+state_x_list[k-30][0,5]+1/2*state_x_list[k-30][0,9])
                Y_30.append(state_x_list[k-30][0,1]+state_x_list[k-30][0,6]+1/2*state_x_list[k-30][0,10])
                Z_30.append(state_x_list[k-30][0,2]+state_x_list[k-30][0,7]+1/2*state_x_list[k-30][0,11])
                X_error_pred_1s.append(abs(state_x_list[k][0,0]-state_x_list[k-30][0,0]+state_x_list[k-30][0,5]+1/2*state_x_list[k-30][0,9]))
    
    
    
    
    plt.figure(1)
    plt.plot(FRAME,X_state,'r',label="Etat")
    plt.plot(FRAME,X_pred,'g',label="prediction Kalman")
    plt.plot(FRAME,detection_X,'ob',label="detections")
    plt.plot(FRAME,X_pred_no_Kalman,'gray',label="sans Kalman")
    plt.plot(FRAME_30,X_30,'purple',label="prediction 1s")
    error_x_pred=sum(X_error_pred_1s)/len(X_error_pred_1s)
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.plot(FRAME,vx_kalman,'r',label="Kalman")
    plt.plot(FRAME,vx_sans_kalman,'b',label="Sans Kalman")
    plt.legend()
    plt.show()
    
    plt.figure(3)
    plt.plot(FRAME,Z_state,'r',label="Etat")
    plt.plot(FRAME,Z_pred,'g',label="prediction Kalman")
    plt.plot(FRAME,detection_Z,'ob',label="detections")
    plt.plot(FRAME,Z_pred_no_Kalman,'gray',label="sans Kalman")
    plt.plot(FRAME_30,Z_30,'purple',label="prediction 1s")
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.plot(FRAME,vz_kalman,'r',label="Kalman")
    plt.plot(FRAME,vz_sans_kalman,'b',label="Sans Kalman")
    plt.legend()
    plt.show()
    
    plt.figure(5)
    plt.plot(FRAME,Y_state,'r',label="Etat")
    plt.plot(FRAME,Y_pred,'g',label="prediction Kalman")
    plt.plot(FRAME,detection_Y,'ob',label="detections")
    plt.plot(FRAME,Y_pred_no_Kalman,'gray',label="sans Kalman")
    plt.plot(FRAME_30,Y_30,'purple',label="prediction 1s")
    plt.legend()
    plt.show()
    return error_x_pred

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



class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,distance,R_list,Q_list):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    T=1/30
    self.kf = KalmanFilter(dim_x=13, dim_z=5)
    self.kf.F = np.array([[1,0,0,0,0,1*T,0,0,0,1/2*T**2,0,0,0],[0,1,0,0,0,0,1*T,0,0,0,1/2*T**2,0,0],[0,0,1,0,0,0,0,1*T,0,0,0,1/2*T**2,0],[0,0,0,1,0,0,0,0,T,0,0,0,1/2*T**2],  [0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1*T,0,0,0],[0,0,0,0,0,0,1,0,0,0,1*T,0,0],[0,0,0,0,0,0,0,1,0,0,0,1*T,0],[0,0,0,0,0,0,0,0,1,0,0,0,T],[0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0]])


    self.kf.R=np.multiply(self.kf.R,R_list)
    self.kf.P[5:,5:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.P[2,2]*=10
    
    self.kf.Q=np.multiply(self.kf.Q,Q_list)


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


class Sort(object):
  def __init__(self,R_list,Q_list,max_age=3,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.R_list=R_list
    self.Q_list=Q_list

  def update(self,dets,distances):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    T=1/30
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    pred=[]
    predicted_state=[]
    state_x=[]
    output=[]
    predicted_distances=[]
    associated_detections=[]
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
        associated_detections.append((dets[d,:][0],distances[d][0],trk.id+1))

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:],distances[i],self.R_list,self.Q_list) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        trk.is_valid(self.min_hits)
        d = trk.get_state()[0]
        current_state=trk.kf.x
        compa_prediction=np.array([[current_state[0]+T*current_state[5]+T**2*1/2*current_state[9]],
                                   [current_state[1]+T*current_state[6]+T**2*1/2*current_state[10]],
                                   [current_state[2]+T*current_state[7]+T**2*1/2*current_state[11]],
                                   [current_state[3]+T*current_state[8]+T**2*1/2*current_state[12]],
                                   [current_state[4]],
                                   [current_state[5]+T*current_state[9]],
                                   [current_state[6]+T*current_state[10]],
                                   [current_state[7]+T*current_state[11]],
                                   [current_state[8]+T*current_state[12]],
                                   [current_state[9]],
                                   [current_state[10]],
                                   [current_state[11]],
                                   [current_state[12]]])[:,:,0]
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
    output.append((ret,pred,predicted_state,state_x,associated_detections))
    return output
