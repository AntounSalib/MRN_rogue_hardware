#!/usr/bin/env python

import numpy as np
import rospy
import json
import time
import math
import copy
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import String
from nod_controller import NodController

from tf.transformations import euler_from_quaternion
from neighbors import sensed_neighbors
from constants import ROBOT_NAMES

class Turtlebot:
    def __init__(self, robot_name, robot_ip):
        self.robot_name = robot_name
        self.robot_ip = robot_ip

        # State variables
        self.prev_time = None
        self.prev_info = None
        self.commanded_velocity = 0.0
        self.info = {
            "name": self.robot_name,
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "heading": 0.0,
        }

        rospy.init_node('tb_controller', anonymous=True)

        # Setup cmd_vel publisher
        self.rate = rospy.Rate(10)
        cmd_vel_channel = f'/{robot_name}/cmd_vel'
        self.vel_pub = rospy.Publisher(cmd_vel_channel, Twist, queue_size=10)

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

        # Initialize NodController
        self.nod_controller = NodController(self.robot_name, time.time())

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

        # --- Compute 2D velocity ---
        if self.info is not None and self.prev_time is not None and self.prev_info is not None:
            dt = current_time - self.prev_time
            prev_pos = self.prev_info["position"]

            if dt > 0:
                vx = (x - prev_pos[0]) / dt
                vy = (y - prev_pos[1]) / dt
                # low-pass filter
                alpha = 1  # smoothing factor
                self.info["velocity"][0] = alpha * vx + (1 - alpha) * self.prev_info["velocity"][0]
                self.info["velocity"][1] = alpha * vy + (1 - alpha) * self.prev_info["velocity"][1]

        # --- Store previous info as a deep copy to avoid pointer issues ---
        self.prev_info = copy.deepcopy(self.info)
        self.prev_time = current_time
        self.info["position"] = [x, y]

        # Use odometry data (quaternion orientation) and convert to euler angles to get robot's heading
        orient_quat = data.transform.rotation
        orient = [orient_quat.x, orient_quat.y, orient_quat.z, orient_quat.w]
        (roll, pitch, yaw) = euler_from_quaternion(orient)

        # Update pos and heading
        self.pos = np.array([x,y])
        self.heading = yaw

        # # computing linear velocities based on commanded velocity and heading
        # self.info["velocity"][0] = self.commanded_velocity * math.cos(self.heading)
        # self.info["velocity"][1] = self.commanded_velocity * math.sin(self.heading)

        # --- Publish info as JSON string ---
        info_msg = String(data=json.dumps(self.info))
        self.info_pub.publish(info_msg)

    def move(self, lin_vel, ang_vel):
        vel_msg = Twist()
        vel_msg.linear.x = lin_vel
        vel_msg.angular.z = ang_vel
        self.vel_pub.publish(vel_msg)
        self.commanded_velocity = lin_vel
        # print(f'published: {vel_msg}')
        
    def neighbor_callback(self, msg):
        """
        Update the dictionary of neighbors with latest info
        """
        msg_dict = json.loads(msg.data)
        self.neighbors[msg_dict["name"]] = msg_dict
        # print(f"Received info from {msg_dict['name']}: pos={msg_dict['position']}, vel={msg_dict['velocity']}, heading={msg_dict['heading']}")


    def run(self):
        """
        TODO call move here where lin_vel is coming from our autonomy controller (NOD, MPC-CBF, GCBF+) and
        ang_vel is set to 0 for this project
        for hardware safety it is advisable to have a low-level cbf screen the autonomy controller
        TODO get other robot's position, speed, do control, and call move.
        """
        v_tar = self.nod_controller.update_opinion(self.info, self.neighbors, time.time())
        ego_pos = np.array(self.info['position'])
        if abs(ego_pos[0]) > 2.5 or abs(ego_pos[1]) > 2.5:
            print(f"{self.robot_name} reached goal at {ego_pos}, stopping.")
            self.move(0.0)
        else:
            self.move(v_tar, 0)
            
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