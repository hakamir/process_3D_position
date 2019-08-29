## *Work in progress*


# Process 3D position

## Introduction

For the ADAPT project (http://adapt-project.com/), the ESIGELEC/IRSEEM is working on the detection of various objects to autonomous mobility of a wheel chair. The aim is to give the ability to see the position of every elements detected by cameras thanks to active vision and Deep Learning algorithms to move to needed objects.

A usecase is to detect a door (then its handle) to approach the patient to help him to open it.

## Description:

This is a ROS package used to put detected objects thanks to Convolutional Neural Network (Yolov3 in our case) and distance estimator using disparity (Real-time Self-adaptative Deep Stereo of MADNet in our project) into a 3D space.

Here's the ROS architecture :

![alt text](https://github.com/hakamir/process_3D_position/blob/master/ROS_diagram.png)

We use different projects from GitHub that are listed below in [Ressources](https://github.com/hakamir/process_3D_position#ressources).

Here's a brief description of each node of the project:

- The *detection* node works with yolov3 through pytorch and provides bounding box positions (x1, y1, x2, y2 format) with class and score related to.

- The *madnet* node works with tensorflow and provide a disparity map used to get the distance of objects.

- The *post_process* node performs a link between bounding boxes and disparity map and convert it into 3D position point. It corrects to position depending the disparity using a mask process. This node is still in construction and another functions to correct distance estimation and box position will be implemented. (histogram correction, Kalman filter...).

- The *areaMaker* node create 3D boxes that represent objects with specific IDs. Every objects added to a box with the same class name might reprensent the same object and it correct it position and existence score. Every objects are placed in a global referential by knowing the position of the camera (T265).

- The *visualizer* node is only used to visualize the position of the 3D boxes with rviz.

## Usage:

The repository is separated in two packages: *yolo_madnet* containing:
- *detection*
- *madnet*
- *post_process*

We can run every nodes with a `roslaunch` command:
- `roslaunch yolo_madnet yolo_madnet.launch`

and *data_processing* contains:
- *areaMaker*
- *visualizer*

Both can be run with:
- `roslaunch data_processing data_processing.launch`

It is also possible to run each node one by one with this following commands:

- Use Realsense package for D435 and T265 cameras: `roslaunch realsense2_camera rs_d400_and_t_265.launch`
- Detection : `python yolo_madnet/scripts/detection.py`
- Distance estimation (MADNet) : `python yolo_madnet/scripts/madnet.py`
- Post-processing : `python yolo_madnet/scripts/post_process.py`
- Create area for objects : `python data_processing/scripts/areaMaker.py`
- Publish area to be visualize in rviz : `python data_processing/scripts/visualizer.py`

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


## Issues and works in progress

Position of objects aren't precise. It depends on the quality of the detector and the behavior of your mobile support.

Moreover, MADNet has never been trained in Indoor situation. So the distance estimation might be often wrong (especially for large distance). By the way, for Indoor context, the results are quite satisfying for now.

At last, the visualization and the selection of interesting and trustable data is not finished and must be highly improved.
