cmake_minimum_required(VERSION 2.8.3)
project(data_processing)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  message_generation
  geometry_msgs
  message_filters
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  CameraMsg.msg
  ObjectMsg.msg
  ObjectsMsg.msg
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs  # Or other packages containing msgs
 )

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(areaMaker_cpp_node src/areaMaker.cpp)
add_library(ObjectCreator src/ObjectCreator.cpp)
target_link_libraries(areaMaker_cpp_node ObjectCreator)
target_link_libraries(areaMaker_cpp_node ${catkin_LIBRARIES})
add_dependencies(areaMaker_cpp_node ${catkin_EXPORTED_TARGETS})
