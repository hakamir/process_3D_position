//#include "../include/areaMaker.h"
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include "yolo_madnet/PointMsg.h"
#include "yolo_madnet/PointsMsg.h"
#include "data_processing/ObjectMsg.h"
#include "data_processing/ObjectsMsg.h"

#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>

using namespace yolo_madnet;
using namespace std;
using namespace message_filters;

/**
The class is a c++ version of areaMaker.py. It is still under construction and
might be used to replace the python version for a more efficient one.
**/
class areaMaker
{

public:

  struct Vector3 {float x, y, z;};
  struct Quaternion {float w, x, y, z;};

  areaMaker()
  {
    subObj.subscribe(nh, "/object/detected", 1);
    subCamPos.subscribe(nh, "/t265/odom/sample", 1);
    sync.reset(new Sync(_SyncPolicy(10), subObj, subCamPos));
    sync->registerCallback(boost::bind(&areaMaker::callback, this, _1, _2));
  }


/*
  void rotateVectorByQuaternion(const struct Vector3& v, const struct Quaternion& q, struct Vector3& vprime)
  {
      // Extract the vector part of the quaternion
      struct Vector3 u{q.x, q.y, q.z};

      // Extract the scalar part of the quaternion
      float s = q.w;

      // Do the math
      vprime = 2.0f * dot(u, v) * u
            + (s*s - dot(u, u)) * v
            + 2.0f * s * cross(u, v);
  }
*/
  struct Vector3 transposeToGlobal(struct Vector3 point, struct Vector3 camPoint, struct Quaternion quaternion)
  {
    struct Vector3 v{0,0,0};
    //areaMaker::rotateVectorByQuaternion(point, quaternion, v);
    v.x += camPoint.x;
    v.y += camPoint.y;
    v.z += camPoint.z;
    return v;
  }

  void callback(const PointsMsgConstPtr& pointsMsg, const nav_msgs::OdometryConstPtr& cameraPosMsg)
  {

    data_processing::ObjectsMsg objects;

    for (size_t i = 0; i < pointsMsg->point.size(); i++)
    {
      PointMsg pt = pointsMsg->point[i];
      // Create a vector for local point
      struct Vector3 localPoint{pt.position.z, - pt.position.x, pt.position.y};

      // We don't want to care about object distance upper 8 meters
      if (localPoint.x > 8.0){
        continue;
      }

      // Get class, score and ID of the object
      string const _class = pt.obj_class;
      double const score = pt.score;
      int long const id = pt.id;

      // Get the scale of the object's box
      // Convert scale referential to camera referential
      struct Vector3 const scale{pt.scale.z, pt.scale.x, pt.scale.y};

      // Get the camera position (euler vector) and rotation (quaternion)
      struct Vector3 const camPoint{cameraPosMsg->pose.pose.position.x,
                              cameraPosMsg->pose.pose.position.y,
                              cameraPosMsg->pose.pose.position.z};

      struct Quaternion const quaternion{cameraPosMsg->pose.pose.orientation.z,
                                    cameraPosMsg->pose.pose.orientation.w,
                                  - cameraPosMsg->pose.pose.orientation.x,
                                    cameraPosMsg->pose.pose.orientation.y};

      // Transpose local_point into global referential
      struct Vector3 globalPoint = areaMaker::transposeToGlobal(localPoint, camPoint, quaternion);

    }
  }
private:

  ros::NodeHandle nh;
  message_filters::Subscriber<PointsMsg> subObj;
  message_filters::Subscriber<nav_msgs::Odometry> subCamPos;
  typedef sync_policies::ApproximateTime<PointsMsg, nav_msgs::Odometry> _SyncPolicy;
  typedef Synchronizer<_SyncPolicy> Sync;
  boost::shared_ptr<Sync> sync;

};


int main(int argc, char** argv){

  ros::init(argc, argv, "areaMaker_cpp_node");

  areaMaker am;

  ros::spin();

  return 0;
}
