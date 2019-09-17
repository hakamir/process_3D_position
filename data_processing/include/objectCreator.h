#include <string>
#include <ctime>
#include <cmath>

class ObjectCreator
{
public:
  struct Vector3 {float x, y, z;};
  struct Quaternion {float w, x, y, z;};
  ObjectCreator(struct Vector3 pointPos, struct Vector3 camPos, struct Vector3 scale, struct Quaternion q, std::string _class, float score, int ID);
  ~ObjectCreator();
  float IoU3D(struct Vector3 scale, struct Vector3 point);
  void calibrate(struct Vector3 point, struct Quaternion q, float score, struct Vector3 scale);
  float logistic(int iteration, float IoU, float score);
  void decaying();

  // Getters
  struct areaMaker::Vector3 getCenter();
  struct areaMaker::Vector3 getScale();
  struct areaMaker::Quaternion getQuaternion();
  std::string getClass();
  float getScore();
  int getID();
  int getIteration();
  float getCreationTime();
  float getLastDetectionTime();

  // Setters
  void setID(int ID);
  void setLastDetectionTime(float time);

private:
  struct areaMaker::Vector3 m_center;
  struct areaMaker::Vector3 m_camPos;
  struct areaMaker::Vector3 m_scale;
  struct areaMaker::Quaternion m_quaternion;
  std::string m_class;
  float m_score;
  int m_ID;
  int m_iteration;
  float m_creationTime;
  float m_lastDetection;
};
