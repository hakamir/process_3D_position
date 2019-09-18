#include "../include/areaMaker.h"

using namespace yolo_madnet;
using namespace std;
using namespace message_filters;

/**
The class is a c++ version of areaMaker.py. It is still under construction and
might be used to replace the python version for a more efficient one.
**/

areaMaker::areaMaker()
{
  subObj.subscribe(nh, "/object/detected", 1);
  subCamPos.subscribe(nh, "/t265/odom/sample", 1);
  sync.reset(new Sync(_SyncPolicy(10), subObj, subCamPos));
  sync->registerCallback(boost::bind(&areaMaker::callback, this, _1, _2));
  pub = nh.advertise<data_processing::ObjectsMsg>("/object/position/3D", 1);
}

/*
This method is used to perform the rotation of the vector v by the Quaternion q
and the result is a rotated vector vprime.
*/
void areaMaker::rotateVectorByQuaternion(const struct Vector3& v, const struct Quaternion& q, struct Vector3& vprime)
{
    float num12 = q.x + q.x;
    float num2 = q.y + q.y;
    float num = q.z + q.z;
    float num11 = q.w * num12;
    float num10 = q.w * num2;
    float num9 = q.w * num;
    float num8 = q.x * num12;
    float num7 = q.x * num2;
    float num6 = q.x * num;
    float num5 = q.y * num2;
    float num4 = q.y * num;
    float num3 = q.z * num;
    float num15 = ((v.x * ((1.0f - num5) - num3)) + (v.y * (num7 - num9))) + (v.z * (num6 + num10));
    float num14 = ((v.x * (num7 + num9)) + (v.y * ((1.0f - num8) - num3))) + (v.z * (num4 - num11));
    float num13 = ((v.x * (num6 - num10)) + (v.y * (num4 + num11))) + (v.z * ((1.0f - num8) - num5));
    vprime.x = num15;
    vprime.y = num14;
    vprime.z = num13;
}
/***
  Description:
  ============
  This function is used to transpose the position of the point into global
  referential based on the tracking camera position.

  Input:
  ------
  - point: The position of the point (x,y,z) based on the camera referential
  - cam_point: The position of the tracking camera T265
  - quatertion: The rotation of the tracking camera T265

  Output:
  -------
  - point: The position of the input point set in global referential
***/
struct Vector3 areaMaker::transposeToGlobal(struct Vector3 point, struct Vector3 camPoint, struct Quaternion quaternion)
{
  struct Vector3 v{0,0,0};
  areaMaker::rotateVectorByQuaternion(point, quaternion, v);
  v.x += camPoint.x;
  v.y += camPoint.y;
  v.z += camPoint.z;
  return v;
}

/***
  Description:
  ============
  Main function of the class.

  Input:
  ------
  - pointMsg: A list of all detected object. Each element is built
  that way:
      * Vector3 position: (x, y, z) position based of camera referential

      * Vector3 scale: 3D box scale (x, y, z) based on the bounding box.

      * string obj_class: The class of the object

      * float32 score: the detection score of the object

  - cameraPosMsg: All the data provided by the Realsense T265 camera.
  The used data here will be position (x,y,z) in meter and the orientation
  (w, i, j, k) given in quaternion.

  Output:
  -------
  - Objects: A list of object containing specific data.
      * Vector3 center:
       -- x: the x position in space (relative to the global position)
       -- y: the y position in space (relative to the global position)
       -- z: the z position in space (relative to the global position)

      * Quaternion rotation: The rotation of the object

      * Vector3 scale
       -- scale_x: the scale in x of the box based on the bounding box
       -- scale_y: the scale in y of the box based on the bounding box
       -- scale_z: the scale in z of the box based on the bounding box

      * float64 creation_time: the creation time of the object

      * float64 last_detection_time: the last detection time of the object

      * string obj_class: The class of the object

      * float32 score: the detection score of the object

      * int32 ID: The unique ID of the object
***/
void areaMaker::callback(const PointsMsgConstPtr& pointsMsg, const nav_msgs::OdometryConstPtr& cameraPosMsg)
{
  data_processing::ObjectsMsg objects;
  bool lockObjCreator = true;
  float iou;

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
    int long const ID = pt.id;

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

    // If no object exists, we create a new mesh at the given position of the added point
    if (m_objList.size() == 0)
    {
      m_objList.push_back(ObjectCreator(globalPoint, camPoint, scale, quaternion, _class, score, ID));
    }

    // Run through each box created
    for (size_t i = 0; i < m_objList.size(); i++)
    {
      // If the class of the item doesn't match with detection, skip
      if (_class != m_objList.at(i).getClass())
      {
        lockObjCreator = false;
        continue;
      }

      else if (ID == m_objList.at(i).getID())
      {
        iou = m_objList.at(i).IoU3D(scale, globalPoint);
        // Perform calibration
        m_objList.at(i).calibrate(globalPoint, quaternion, score, scale);
        lockObjCreator = true;
        // Be careful!!!
        // The ID sent is NOT the ID provided by Sort (cause it is
        // unstable). So, we send for rviz the index of the item in
        //the item list because it is unique.
        objects.object.push_back(joinData(m_objList.at(i), i));
        break;
      }
      else
      {
        iou = m_objList.at(i).IoU3D(scale, globalPoint);
        if (iou > 0)
        {
          // Perform calibration
          m_objList.at(i).calibrate(globalPoint, quaternion, score, scale);
          // We change the ID of the object. The previous one might
          // has been lost by Sort for major cases.
          m_objList.at(i).setID(ID);

          // PROBLEM WITH ITERATION... Stay at '1', no incrementation.


          // Say that no object must be created
          lockObjCreator = true;
          objects.object.push_back(joinData(m_objList.at(i), i));
          break;
        }
        else
        {
          lockObjCreator = false;
        }
      }
      if (time(0) - m_objList.at(i).getLastDetectionTime() > 50)
      {
        m_objList.erase(m_objList.begin() + i);
      }
    }
    if (!lockObjCreator)
    {
      m_objList.push_back(ObjectCreator(globalPoint, camPoint, scale, quaternion, _class, score, ID));
    }
  }
  ros::Time now = ros::Time::now();
  objects.header.stamp = now;
  pub.publish(objects);
}

/***
  Description:
  ============
  Use this to set all data regarding the item in an object ROS message.
  It set header, position and rotation, class, existence probability
  score, id and time details on the message.

  Input:
  ------
  - item: an object from ObjectCreator class.
  - id: The id of the object. Not the Sort one to avoid misinformation.

  Output:
  -------
  - object: a ROS type message from ObjectMsg.msg. It is used to be
  appended in ROS type message from ObjectsMsg.msg.
***/
data_processing::ObjectMsg areaMaker::joinData(ObjectCreator item, int id)
{
  // get all data Publish the position of the object
  struct Vector3 msg_center = item.getCenter();
  struct Vector3 msg_scale = item.getScale();
  struct Quaternion msg_rotation = item.getQuaternion();
  string msg_class = item.getClass();
  float msg_score = item.getScore();
  int msg_ID = id;
  float msg_creation_time = item.getCreationTime();
  float msg_last_detection_time = item.getLastDetectionTime();
  ros::Time now = ros::Time::now();
  data_processing::ObjectMsg object;
  object.header.stamp = now;
  object.center.x = msg_center.x;
  object.center.y = msg_center.y;
  object.center.z = msg_center.z;
  object.scale.x = msg_scale.x;
  object.scale.y = msg_scale.y;
  object.scale.z = msg_scale.z;
  object.rotation.w = msg_rotation.w;
  object.rotation.x = msg_rotation.x;
  object.rotation.y = msg_rotation.y;
  object.rotation.z = msg_rotation.z;
  object.creation_time = msg_creation_time;
  object.last_detection_time = msg_last_detection_time;
  object.obj_class = msg_class;
  object.score = msg_score;
  object.ID = msg_ID;
  return object;
}

int main(int argc, char** argv){

  ros::init(argc, argv, "areaMaker_cpp_node");

  areaMaker am;

  ros::spin();

  return 0;
}
