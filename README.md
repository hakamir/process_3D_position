## *Work in progress*


# Process 3D position

This is a ROS package used to put detected objects thanks to Convolutional Neural Network (Yolov3 in our case) and distance estimator using disparity (Real-time Self-adaptative Deep Stereo of MADNet in our project) into a 3D space. 

Here's the ROS architecture :

![alt text](https://github.com/hakamir/process_3D_position/blob/master/ROS_diagram_2.png)

We use different project from GitHub that are listed below:

## Usage:

For now, no usable roslaunch have been created. Run every scripts by hand in different terminal to proceed. 

## Downloads:

The following link contains the RTSADS and yolov3 repositories of this projet. Just copy them at the right place to get the functions needed for algorithms.

https://drive.google.com/open?id=1iT88gagBGAoxw0BVj_g25YovM_3-SftN

## Ressources:

### For the object detection: 
https://github.com/ultralytics/yolov3
[![DOI](https://zenodo.org/record/2672652#.XPZ-IxixX8s)](http://dx.doi.org/10.5281/zenodo.2675652)

### For the distance estimation: 
https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo
@InProceedings{Tonioni_2019_CVPR,
    author = {Tonioni, Alessio and Tosi, Fabio and Poggi, Matteo and Mattoccia, Stefano and Di Stefano, Luigi},
    title = {Real-time self-adaptive deep stereo},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}

### To use D435 and T265 cameras from Intel(R) in ROS:
https://github.com/IntelRealSense/realsense-ros

### To calibrate the data:
https://github.com/abewley/sort
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}

