#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, TransformStamped
import time

class Turtlebot:
    def __init__(self, robot_name, robot_ip):
        self.robot_name = robot_name
        self.robot_ip = robot_ip

        self.pos = [0, 0]

        rospy.init_node('tb_controller', anonymous=True)
        # Setup cmd_vel publisher
        self.rate = rospy.Rate(10)
        cmd_vel_channel = f'/{robot_name}/cmd_vel'
        self.vel_pub = rospy.Publisher(cmd_vel_channel, Twist, queue_size=10)
        
        # Setup Vicon position subscriber
        vicon_channel = f'/vicon/{robot_name}/{robot_name}'
        self.pose_sub = rospy.Subscriber(vicon_channel, TransformStamped, self.pose_callback)

        rospy.on_shutdown(self.tb_stop)

    def tb_stop(self):
        vel_msg = Twist()

        vel_msg.linear.x = 0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0

        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0    
        vel_msg.angular.z = 0      
        
        self.vel_pub.publish(vel_msg)
        self.rate.sleep()

    def pose_callback(self, data):
        x = data.transform.translation.x
        y = data.transform.translation.y
        self.pos = [x, y]
        # print(f'pose: {x},{y}')

    def move(self, lin_vel, ang_vel):
        vel_msg = Twist()
        vel_msg.linear.x = lin_vel
        vel_msg.angular.z = ang_vel
        self.vel_pub.publish(vel_msg)
        # print(f'published: {vel_msg}')
        

    def run(self):
        """
        TODO call move here where lin_vel is coming from our autonomy controller (NOD, MPC-CBF, GCBF+) and
        ang_vel is set to 0 for this project
        for hardware safety it is advisable to have a low-level cbf screen the autonomy controller
        TODO get other robot's position, speed, do control, and call move.
        """
        self.move(0.05, 0.1)
        self.rate.sleep()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Turtlebot controller")
    parser.add_argument('-robot_name', type=str, required=True)
    parser.add_argument('-robot_ip', type=str, required=True)

    args, unknown = parser.parse_known_args()

    print("Starting", args.robot_name)
    tb = Turtlebot(args.robot_name, args.robot_ip)

    time.sleep(10)

    try:
        while not rospy.is_shutdown():
            tb.run()
        #If we press ctrl + C, the node will stop.
        rospy.spin()
    except rospy.ROSInterruptException:
        pass