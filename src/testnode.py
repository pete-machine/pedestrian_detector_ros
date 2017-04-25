#!/usr/bin/env python
import rospy

from std_msgs.msg import Float64MultiArray

class MyTestNode:
    def __init__(self):
        self.pub = rospy.Publisher("/obstacles",Float64MultiArray)
        self.data= []
        self.msg = Float64MultiArray()
        self.msg.data= [4,0.1,7,0.3,10,0.5]
        self.timer = rospy.Timer(rospy.Duration(0.1),self.on_timer)

        pass
    def on_timer(self,event):
        print "publishing"
        self.msg.data[0] = self.msg.data[0] - 0.1
        self.msg.data[2] = self.msg.data[4] - 0.1
        self.msg.data[4] = self.msg.data[4] - 0.1
        if self.msg.data[0] < 2:
            self.msg.data[0] = 10
        if self.msg.data[2] < 2:
            self.msg.data[2] = 10
        if self.msg.data[4] < 2:
            self.msg.data[4] = 10
        self.pub.publish(self.msg)
        pass


if __name__ == "__main__":
    rospy.init_node("mynode")
    node = MyTestNode()

    rospy.spin()
