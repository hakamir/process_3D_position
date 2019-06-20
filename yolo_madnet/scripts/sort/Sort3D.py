
"""
Created on Mon Jun 17 11:20:41 2019

@author: antoine
"""



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


import os.path
import numpy as np
import time
import argparse
import cv2
from utils.object_segmentation import segmentation
from utils.utils import get_list,plot_one_box,convert_x_to_bbox,plot,Sort



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--output', dest='output', default='output/run_RS_acc/',action='store_true')
    parser.add_argument('--plot', dest='plot', default=True)
    parser.add_argument('--disp_path', dest='disp_path', default="/media/antoine/\
Disque dur portable/Detection+Depth/Yolov3_Stereo_depth_multithread/\
output/Mesure_erreur_GTv3/MAD/disparities_npy/")

    parser.add_argument('--img_path', dest='img_path', default="/media/\
antoine/Disque dur portable/Datasets/RealSense/python/v2/frames/1/")

    parser.add_argument('--detection_path', dest='detection_path', default=
"/media/antoine/Disque dur portable/Detection+Depth/\
Yolov3_Stereo_depth_multithread/output/Mesure_erreur_GTv3/MAD/estimation_npy/")

    args = parser.parse_args()
    return args


def tracking(R_list,Q_list):
      args = parse_args()
      file_list,img_list,disp_list=get_list(args.detection_path,args.img_path,args.disp_path)
      if not os.path.exists(args.output):
            os.makedirs(args.output)
      if not os.path.exists(args.output+"predictions/"):
            os.makedirs(args.output+"predictions/")
      total_time = 0.0
      total_frames = 0
      colours = np.random.rand(1000,3) #used only for display
      colours*=255




      mot_tracker = Sort(R_list,Q_list) #create instance of the SORT tracker
      with open(args.output+'out.txt','w') as out_file:
          history_state_x=[]
          history_prediction=[]
          dets_history=[]
          history_gt=[]
          for frame in range(len(file_list)):
            original_img=cv2.imread(img_list[frame])
            data=np.load(file_list[frame])
            disparity=np.load(disp_list[frame])
            masks,original_img=segmentation(data,disparity,img_list[frame])
            if len(data)>0:
                dets=masks[:,1:5]
                distances=data[:,4]
            else:
                dets=[]
                distances=[]
            total_frames += 1



            start_time = time.time()
            trackers,predictions,predicted_state,state_x,associated_detections=np.array(mot_tracker.update(dets,distances)[0])
            np.save(args.output+"predictions/pred%06i.npy"%frame,[trackers,predictions,predicted_state,state_x,associated_detections])
            if len(trackers)>0:
                trackers=np.concatenate(trackers)
            if len(predicted_state)>0:
                predicted_state=np.concatenate(predicted_state)
            if len(associated_detections)>0:
                associated_detections=np.concatenate(associated_detections)
            if len(predictions)>0:
                predictions=np.concatenate(predictions)
            if len(state_x)>0:
                state_x=np.concatenate(state_x)
            cycle_time = time.time() - start_time
            total_time += cycle_time


            history_prediction.append(predicted_state)
            history_state_x.append(state_x)
            dets_history.append(associated_detections)

            for d in trackers:
                x1,y1,x2,y2=int(d[0]),int(d[1]),int(d[2]),int(d[3])
                label=str(d[4])
                plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(d[4])-1])

                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)


            for d in predictions:
              x1,y1,x2,y2=int(d[0]),int(d[1]),int(d[2]),int(d[3])
              label=str(d[4])+' prediction'
              plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(d[4])-1])

              print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)


            for d in predicted_state:
                idx=d[-1]


                x=d[0:5]
                pixels=convert_x_to_bbox(x)[0]
                x1,y1,x2,y2=int(pixels[0]),int(pixels[1]),int(pixels[2]),int(pixels[3])
                label=str(int(idx))+' prediction'
#                plot_one_box([x1, y1, x2, y2], original_img, label=label, color=colours[int(idx)+100])
                pixel_coord=[int((x2+x1)/2),int((y2+y1)/2)]
                if len(trackers)>0:
                    row=np.where(trackers[:,4]==idx)
                    if(len(row[0]))>0:
                        x1,y1,x2,y2=int(trackers[row[0][0],0]),int(trackers[row[0][0],1]),int(trackers[row[0][0],2]),int(trackers[row[0][0],3])

                        original_position=[int((x2+x1)/2.),int((y2+y1)/2.)]
                        cv2.arrowedLine(original_img,(original_position[0],
                                                      original_position[1]),(pixel_coord[0],pixel_coord[1]),
                                                      colours[int(idx)-1,:],thickness = 2)
                        vitesses=[d[5],d[6],d[7]]
                        vitesse_globale=np.sqrt(vitesses[0]**2+vitesses[1]**2+vitesses[2]**2)
                        vitesse_globale=np.sqrt(vitesses[0]**2+vitesses[2]**2)
                        label="vx:%.3f m/s   vy:%.3f m/s   vz:%.3f m/s   v:%.3f m/s"%(vitesses[0],vitesses[1],vitesses[2],vitesse_globale)
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

            cv2.imwrite(args.output+'out%06i.png'%frame,original_img)
      return history_state_x,history_prediction,history_gt,dets_history


args = parse_args()
R_list=np.array([1,100,10,100,100])
Q_list=np.array([1,1,1,0.1,0.1,0.01,0.01,0.01,0.0001,0.0001,0.0001,0.0001,0.00001])
state_x_list,prediction_list,GT,detections=tracking(R_list,Q_list)

if args.plot:
    plot(state_x_list,prediction_list,detections)
