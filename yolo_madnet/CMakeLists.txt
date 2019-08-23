cmake_minimum_required(VERSION 2.8.3)
project(yolo_madnet)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  geometry_msgs
  cv_bridge
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  BboxMsg.msg
  PointMsg.msg
  PointsMsg.msg
  DetectionMsg.msg
  DisparityMsg.msg
)


## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   sensor_msgs
   geometry_msgs
   std_msgs  # Or other packages containing msgs
 )



catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

