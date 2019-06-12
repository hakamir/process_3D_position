## *Work in progress*


# Process 3D position

# Introduction

For the ADAPT project (http://adapt-project.com/), the ESIGELEC/IRSEEM is working on the detection of various objects to autonomous mobility of a wheel chair. The aim is to give the ability to see the position of every elements detected by cameras thanks to active vision and Deep Learning algorithms to move to needed objects. 

A usecase is to detect a door (then it handle) to approach the patient to help him to open it. 

## Description:

This is a ROS package used to put detected objects thanks to Convolutional Neural Network (Yolov3 in our case) and distance estimator using disparity (Real-time Self-adaptative Deep Stereo of MADNet in our project) into a 3D space. 

Here's the ROS architecture :

![alt text](https://github.com/hakamir/process_3D_position/blob/master/ROS_diagram_2.png)

We use different project from GitHub that are listed below in Ressources.

Here's a brief description of each node of the project:

- The *Detection* node works with yolov3 through pytorch and provides bounding box positions (x1, y1, x2, y2 format) with class and score related to. 

- The *madnet* node works with tensorflow and provide a disparity map used to get the distance of objects.

- The *post_process* node performs a link between bounding boxes and disparity map. This node is still in construction and another functions to correct distance estimation and box position will be implemented. (mask, histogram correction...). It currently returns the center of the bounding box position in pixel (u,v) with the calculate distances in meter (with the class and score associated). 

- The *spacialize* node convert u, v and z position to a x, y, z position, all in meters. This node will implement a Kalman corrector to perform better performance (using SORT). 

- The *areaMaker* node create 3D boxes that represent objects with specific IDs. Every objects added to a box with the same class name might reprensent the same object and it correct it position and existence score. Every objects are placed in a global referential by knowing the position of the camera (T265). 

- The *animated3Dplot* node is only used to visualize the position of the 3D boxes. This is currently a very basic program that will be upgrade later for a better utilisation. 

## Usage:

For now, no usable roslaunch have been created. Run every scripts by hand in different terminal to proceed. 

- Use Realsense package for D435 cameras: `roslaunch realsense2_camera rs_camera.launch`
- Use T265 data: `python data_processing/scripts/t265.py`
- Detection : `python yolo_madnet/scripts/detection.py`
- Distance estimation (MADNet) : `python yolo_madnet/scripts/madnet.py`
- Post-processing : `python yolo_madnet/scripts/post_process.py`
- Spacialize : `python data_processing/scripts/spacialize.py`
- Create area for objects : `python data_processing/scripts/areaMaker.py`
- Show area in OpenGL viewer : `python data_processing/scripts/animated3Dplot.py`

## Requirements:

- For detection node:
`torch`

- For depth estimation node:
`tensorflow-gpu`

- For realsense camera:
`pyrealsense2`

- For image management:
`numpy`
`cv2`

- For OpenGL node:
`pyqtgraph`
`pyopengl`
Install PyQt5 with `sudo python -m pip install PyQt5` if asked. 

- For areaMaker node:
`pyquaternion`
`colorama`

To install all the requirements: 
`sudo python -m pip install -r requirement.txt`

Tensorflow is working and was compiled and tested with 1.13.1 GPU version through cuda 10.0 and CuDNN 7.5 with Python 2.7 on Ubuntu 16.04 LTS. 

Pytorch works on 1.1.0 version with Python 2.7.

## Downloads:

The following link contains the RTSADS (Real-Time Self-Adaptative Deep-Stereo) and yolov3 repositories of this projet. Just copy them at the right place to get the functions needed for algorithms (path : *process_3D_position/yolo_madnet/scripts*).

https://drive.google.com/open?id=1iT88gagBGAoxw0BVj_g25YovM_3-SftN

## Ressources:

### For the object detection: 
https://github.com/ultralytics/yolov3

### For the distance estimation: 
https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo

### To use D435 and T265 cameras from Intel(R) in ROS:
https://github.com/IntelRealSense/realsense-ros

### To calibrate the data:
https://github.com/abewley/sort

*Caution : This program has been highly modified to fit to our context, meaning process to 3D positions.*
