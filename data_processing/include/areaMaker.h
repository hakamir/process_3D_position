#ifndef AREAMAKER
#define AREAMAKER

// ROS utilities
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// ROS messages
#include <nav_msgs/Odometry.h>
#include "yolo_madnet/PointMsg.h"
#include "yolo_madnet/PointsMsg.h"
#include "data_processing/ObjectMsg.h"
#include "data_processing/ObjectsMsg.h"

// Customs headers
#include "../include/ObjectCreator.h"
#include "../include/vectors.h"

#include <vector>


class areaMaker
{
  public:

    areaMaker();
    void rotateVectorByQuaternion(const struct Vector3& v, const struct Quaternion& q, struct Vector3& vprime);
    struct Vector3 transposeToGlobal(struct Vector3 point, struct Vector3 camPoint, struct Quaternion quaternion);
    void callback(const yolo_madnet::PointsMsgConstPtr& pointsMsg, const nav_msgs::OdometryConstPtr& cameraPosMsg);
    data_processing::ObjectMsg joinData(ObjectCreator item, int id);

  private:

    ros::NodeHandle nh;
    message_filters::Subscriber<yolo_madnet::PointsMsg> subObj;
    message_filters::Subscriber<nav_msgs::Odometry> subCamPos;
    typedef message_filters::sync_policies::ApproximateTime<yolo_madnet::PointsMsg, nav_msgs::Odometry> _SyncPolicy;
    typedef message_filters::Synchronizer<_SyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;

    ros::Publisher pub;

    std::vector<ObjectCreator> m_objList;

};

#endif
