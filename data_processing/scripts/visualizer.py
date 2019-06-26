import rospy
from visualization_msgs.msg import Marker
from data_processing.msg import ObjectsMsg
import random

class visualizer:
    def __init__(self):
        rospy.init_node('visualizer')
        rospy.Subscriber('/object/position/3D', ObjectsMsg, self.process, queue_size=10)
        self.pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
        self.classes = self.load_classes('/home/zhaoqi/catkin_ws/src/process_3D_position/yolo_madnet/scripts/yolov3/data/coco.names')
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        rospy.spin()

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def process(self, msg):
        for object in msg.object:
            marker = Marker()
            marker.header.frame_id = "/my_frame"
            #marker.header.stamp = rospy.now()
            print(object)
            marker.ns = object.obj_class
            marker.id = object.ID

            marker.type = 1

            marker.action = 0

            marker.pose.position.x = object.center.x
            marker.pose.position.y = object.center.y
            marker.pose.position.z = object.center.z
            marker.pose.orientation.x = object.rotation.x
            marker.pose.orientation.y = object.rotation.y
            marker.pose.orientation.z = object.rotation.z
            marker.pose.orientation.w = object.rotation.w

            marker.scale.x = object.scale.x
            marker.scale.y = object.scale.y
            marker.scale.z = object.scale.z
            for cls in self.classes:
                if object.obj_class == cls:
                    color_index = self.classes.index(cls)
            print(self.colors[color_index])
            marker.color.r = float(self.colors[color_index][0])/255
            marker.color.g = float(self.colors[color_index][1])/255
            marker.color.b = float(self.colors[color_index][2])/255
            marker.color.a = object.score
            print(marker.color.r, marker.color.g, marker.color.g, marker.color.a)

            marker.lifetime = rospy.Duration()
            self.pub.publish(marker)

if __name__ == '__main__':
    try:
        visualizer()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualizer node.')
