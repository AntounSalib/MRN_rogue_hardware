#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import String
import json
import time
import math

from agent_info import AgentInfo
from constants import ROBOT_NAMES

class Turtlebot:
    def __init__(self, robot_name, robot_ip):
        self.robot_name = robot_name
        self.robot_ip = robot_ip

        # State variables
        self.prev_pos = None
        self.pos = [0, 0]

        rospy.init_node('tb_controller', anonymous=True)

        # # Setup cmd_vel publisher
        # self.rate = rospy.Rate(10)
        # cmd_vel_channel = f'/{robot_name}/cmd_vel'
        # self.vel_pub = rospy.Publisher(cmd_vel_channel, Twist, queue_size=10)

        # Create publisher to send velocity and heading
        vel_heading_str = '/' + self.robot_name + '/info'
        self.info_pub = rospy.Publisher(vel_heading_str, String, queue_size = 10)   
        
        # Setup Vicon position subscriber
        vicon_channel = f'/vicon/{robot_name}/{robot_name}'
        self.pose_sub = rospy.Subscriber(vicon_channel, TransformStamped, self.pose_callback)

        # Subscribe to other robots
        self.neighbors = {}
        for other_robot in ROBOT_NAMES:
            if other_robot != self.robot_name:
                topic_name = f'/{other_robot}/info'
                rospy.Subscriber(topic_name, String, self.neighbor_callback)


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
        current_time = data.header.stamp.to_sec()  # ROS time

        # --- Compute 2D velocity (same as before) ---
        if self.prev_pos is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                vx = (x - self.prev_pos[0]) / dt
                vy = (y - self.prev_pos[1]) / dt
                self.vel = [vx, vy]

                # low-pass filter
                alpha = 0.5  # smoothing factor
                self.vel[0] = alpha * vx + (1-alpha) * self.vel[0]
                self.vel[1] = alpha * vy + (1-alpha) * self.vel[1]

        self.prev_pos = [x, y]
        self.prev_time = current_time
        self.pos = [x, y]

        # --- Extract heading (yaw) from quaternion ---
        q = data.transform.rotation
        # Quaternion components
        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w

        # Yaw formula for 2D (rotation around Z axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)  # in radians

        self.heading = yaw  # store heading

        ego_info = AgentInfo(
            name=self.robot_name,
            position=self.pos,
            velocity=self.vel,
            heading=self.heading
        )

        AgentInfo_msg = String(data=json.dumps(ego_info.__dict__))
        self.info_pub.publish(AgentInfo_msg)

    def move(self, lin_vel, ang_vel):
        vel_msg = Twist()
        vel_msg.linear.x = lin_vel
        vel_msg.angular.z = ang_vel
        self.vel_pub.publish(vel_msg)
        # print(f'published: {vel_msg}')
        
    def neighbor_callback(self, msg):
        """
        Update the dictionary of neighbors with latest info
        """
        msg_dict = json.loads(msg.data)
        self.neighbors[msg_dict["name"]] = msg
        # Optional: print for debugging
        print(f"Received info from {msg_dict['name']}: pos={msg_dict['position']}, vel={msg_dict['velocity']}, heading={msg_dict['heading']}")


    def run(self):
        """
        TODO call move here where lin_vel is coming from our autonomy controller (NOD, MPC-CBF, GCBF+) and
        ang_vel is set to 0 for this project
        for hardware safety it is advisable to have a low-level cbf screen the autonomy controller
        TODO get other robot's position, speed, do control, and call move.
        """
        print(f"{self.robot_name=},{self.pos=}, {self.vel=}, {self.heading=}")

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