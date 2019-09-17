#include "../include/objectCreator.h"

using namespace std;

ObjectCreator::ObjectCreator(struct Vector3 pointPos, struct Vector3 camPos, struct Vector3 scale, struct Quaternion q, string _class, float score, int ID)
{
  m_center = pointPos;
  m_camPos = camPos;
  m_scale = scale;
  m_quaternion = q;
  m_class = _class;
  m_score = score;
  m_ID = ID;
  m_iteration = 1;
  m_creationTime = time(0);
  m_lastDetection = m_creationTime;
}

ObjectCreator::~ObjectCreator()
{

}

/***
  Description:
  ============
  This method tries to catch matching between the input scale and the
  scale of the object. It is used to indicate if the provided box can
  correspond to this object.

  Input:
  ------
  - scale: a scale of a new detected object.
  - point: the point in space of the new detected object. It must be in
  global environment.

  Output:
  -------
  - IoU: The Intersection of Union is returned (value between 0 and 1). It
  corresponds to the IoU between the object box and the entry box (with
  scale and center point).
***/
float ObjectCreator::IoU3D(struct Vector3 scale, struct Vector3 point)
{

  // Calculate the area of both boxes
  area = m_scale.x * m_scale.y * m_scale.z
  new_area = scale.x * scale.y * scale.z
  ratio = area / new_area

  // Retrieve the initial (x1,x2,y1,y2,z1,z2) positions of the new box
  x1 = point.x - scale.x/2
  x2 = point.x + scale.x/2
  y1 = point.y - scale.y/2
  y2 = point.y + scale.y/2
  z1 = point.z - scale.z/2
  z2 = point.z + scale.z/2

  // Retrieve the initial (x1,x2,y1,y2,z1,z2) positions of the actual box
  x1p = m_center.x - m_scale.x/2
  x2p = m_center.x + m_scale.x/2
  y1p = m_center.y - m_scale.y/2
  y2p = m_center.y + m_scale.y/2
  z1p = m_center.z - m_scale.z/2
  z2p = m_center.z + m_scale.z/2

  // calculate the area overlap with the previous data
  overlap = max((min(x2,x2p)-max(x1,x1p)),0)*max((min(y2,y2p)-max(y1,y1p)),0)*max((min(z2,z2p)-max(z1,z1p)),0)

  // calculate the IoU with the previous calculated area
  IoU = overlap / (area + new_area - overlap)

  return IoU
}

/***
  Description:
  ============
  The following function set the new position of the object based on a
  given point. It also updates the existence probability score, the scale
  of the object, increase the detection iteration by one, set the rotation
  of the object on the tracking camera quaternion value and set the last
  detection time.

  Input:
  ------
  - point: the new position of the object
  - cam_pos: the position of the tracking camera T265
  - quaternion: the rotation of the tracking camera T265
  - score: the detection score of the object
  - scale: the input scale of the object (x, y, z)
***/
void ObjectCreator::calibrate(struct Vector3 point, struct Quaternion q, float score, struct Vector3 scale)
  {
    m_quaternion = q;
    float IoU = ObjectCreator::IoU3D(scale, point);
    m_scale = scale;
    m_center = point;
    m_iteration += 1;

    // Lock to 1000 to avoid overcomsumption of memory
    if (m_iteration > 1000)
    {
      m_iteration = 1000;
    }
    // Set the score depending iteration, IoU and detection score
    // Be careful when IoU is equal to zero!
    m_score = ObjectCreator::logistic(m_iteration, IoU, m_score);
    m_lastDetection = time(0);
  }
)

/***
  Description:
  ============
  A logistic function use to provide the existence probability score of
  the object depending the entry.

  Input:
  ------
  - iteration: The number of detection of the object (in frame)
  - IoU: The IoU returned between the entry box and the self box
  (see iou_3D method)
  - score: The detection score


  Output:
  -------
  - The existence probability score of the object depending the parameters
***/
float ObjectCreator::logistic(int iteration, float IoU, float score)
{
  return (2.0 / pi) * atan((iteration - 1) * IoU * score);
}



// Getters
struct Vector3 ObjectCreator::getCenter()
{
  return m_center;
}
struct Vector3 ObjectCreator::getScale()
{
  return m_scale;
}
struct Quaternion ObjectCreator::getQuaternion()
{
  return m_quaternion;
}
std::string ObjectCreator::getClass()
{
  return m_class;
}
float ObjectCreator::getScore()
{
  return m_score;
}
int ObjectCreator::getID()
{
  return m_ID;
}
int ObjectCreator::getIteration()
{
  return m_iteration;
}
float ObjectCreator::getCreationTime()
{
  return m_creationTime;
}
float ObjectCreator::getLastDetectionTime()
{
  return m_lastDetection;
}

// Setters
void ObjectCreator::setID(int ID)
{
  m_ID = ID;
}
void ObjectCreator::setLastDetectionTime()
{
  m_lastDetection = time(0);
}
