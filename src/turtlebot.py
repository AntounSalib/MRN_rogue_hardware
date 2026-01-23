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
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion
from neighbors import sensed_neighbors
from constants import ROBOT_NAMES, EPS, NodConfig, ROGUE_AGENTS

from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState
from gazebo_msgs.msg import ModelState

class Turtlebot:
    def __init__(self, robot_name, robot_ip, sim, x_init, y_init, z_init, yaw_init, simulation_on):
        self.robot_name = robot_name
        # if robot_ip is provided, use it
        if robot_ip is not None:
            self.robot_ip = robot_ip
        self.simulation_on = simulation_on

        # State variables
        self.prev_time = None
        self.commanded_velocity = 0.0
        self.info = {
            "name": self.robot_name,
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "heading": 0.0,
        }

        if sim:
            pos_init = np.array([x_init, y_init])
            self.pos = np.asarray(pos_init, dtype=float)
            heading_init = yaw_init
            self.info["position"] = [float(x_init), float(y_init)]
            self.info["heading"] = float(heading_init)
        else:
            self.pos = None

        rospy.init_node('tb_controller', anonymous=True)

        # Setup cmd_vel publisher
        self.rate = rospy.Rate(10)
        cmd_vel_channel = f'/{robot_name}/cmd_vel'
        self.vel_pub = rospy.Publisher(cmd_vel_channel, Twist, queue_size=10)

        # Create publisher to send velocity and heading
        vel_heading_str = '/' + self.robot_name + '/info'
        self.info_pub = rospy.Publisher(vel_heading_str, String, queue_size = 10)   
        
        if not sim:
            # Setup Vicon position subscriber
            vicon_channel = f'/vicon/{robot_name}/{robot_name}'
            self.pose_sub = rospy.Subscriber(vicon_channel, TransformStamped, self.pose_callback)

            # Subscribe to other robots
            self.neighbors = {}
            for other_robot in ROBOT_NAMES:
                if other_robot != self.robot_name:
                    topic_name = f'/{other_robot}/info'
                    rospy.Subscriber(topic_name, String, self.neighbor_callback)
        else:
            odom_str = '/' + robot_name + '/odom'
            
            # Subscribe to the odometry topic '/odom' to receive odometry data
            self.pose_sub = rospy.Subscriber(odom_str, Odometry, self.sim_pose_callback)
            
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

    def sim_pose_callback(self, data):
        x = float(data.pose.pose.position.x)
        y = float(data.pose.pose.position.y)

        orient_quat = data.pose.pose.orientation
        orient = [orient_quat.x, orient_quat.y, orient_quat.z, orient_quat.w]
        (_, _, yaw) = euler_from_quaternion(orient)

        # Fill the same info structure used in hardware mode
        self.info["position"] = [x, y]
        self.info["heading"] = yaw

        # Get local velocity
        v_linear = data.twist.twist.linear.x
        v_lateral = data.twist.twist.linear.y # usually 0 for non-holonomic

        # Rotate to World Frame
        vx_global = v_linear * math.cos(yaw) - v_lateral * math.sin(yaw)
        vy_global = v_linear * math.sin(yaw) + v_lateral * math.cos(yaw)

        self.info["velocity"] = [vx_global, vy_global]

        # # Use odom twist as velocity (in the odom frame)
        # vx = float(data.twist.twist.linear.x)
        # vy = float(data.twist.twist.linear.y)
        # self.info["velocity"] = [vx, vy]

        # Publish so neighbors can subscribe
        self.info_pub.publish(String(data=json.dumps(self.info)))


    def pose_callback(self, data):
        x = data.transform.translation.x
        y = data.transform.translation.y
        current_time = data.header.stamp.to_sec()  # ROS time

        prev_info = copy.deepcopy(self.info)

        # --- Compute 2D velocity ---
        if prev_info is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            prev_pos = self.info["position"]

            if dt > 0:
                vx = (x - prev_pos[0]) / dt
                vy = (y - prev_pos[1]) / dt
                # low-pass filter
                alpha = .3  # smoothing factor
                self.info["velocity"][0] = alpha * vx + (1 - alpha) * prev_info["velocity"][0]
                self.info["velocity"][1] = alpha * vy + (1 - alpha) * prev_info["velocity"][1]

        # --- Store previous info as a deep copy to avoid pointer issues ---
        self.prev_time = current_time
        self.info["position"] = [x, y]

        # Use odometry data (quaternion orientation) and convert to euler angles to get robot's heading
        orient_quat = data.transform.rotation
        orient = [orient_quat.x, orient_quat.y, orient_quat.z, orient_quat.w]
        (roll, pitch, yaw) = euler_from_quaternion(orient)
        self.info["heading"] = yaw

        # computing linear velocities based on commanded velocity and heading
        vel_mag = math.hypot(self.info["velocity"][0], self.info["velocity"][1]) + EPS
        self.info["velocity"][0] = vel_mag * math.cos(self.info["heading"])
        self.info["velocity"][1] = vel_mag * math.sin(self.info["heading"])

        # --- Publish info as JSON string ---
        info_msg = String(data=json.dumps(self.info))
        self.info_pub.publish(info_msg)

        # print(f"{self.robot_name} pos: {self.info['position']}, vel: {self.info['velocity']}, heading: {self.info['heading']:.2f}")

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
 

    def _get_v_commanded(self, v_target):
        v_current = np.linalg.norm(self.info["velocity"])

        def _v_dot_rhs(v_val: float) -> float:
            return  NodConfig.kin.KAPPA_V*(v_target - v_val)
        
        
        # Iterate RK4 updates on the rogueness score until it stabilizes
        time_step = 0.1
        for _ in range(1):
            k1 = _v_dot_rhs(v_current)
            k2 = _v_dot_rhs(v_current + 0.5 * time_step * k1)
            k3 = _v_dot_rhs(v_current + 0.5 * time_step * k2)
            k4 = _v_dot_rhs(v_current + time_step * k3)

            next_v_current = v_current + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if abs(next_v_current - v_current) < 1e-4:
                v_current = next_v_current
                break
            v_current = next_v_current

        return v_current

    def run(self):
        ego_pos = self.info['position']
        if (abs(ego_pos[0]) > 5 or abs(ego_pos[1]) > 5 or (ego_pos[1]) < -5):
            # print(f"{self.robot_name} reached goal at {ego_pos}, stopping.")
            self.move(0, 0)
            return


        if self.robot_name in ROGUE_AGENTS:
            self.move(NodConfig.kin.V_NOMINAL, 0)
            return

        v_tar = self.nod_controller.update_opinion(self.info, self.neighbors, time.time())
        v_command = self._get_v_commanded(v_tar)
        self.move(v_command, 0)

        self.rate.sleep()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Turtlebot controller")
    parser.add_argument('-robot_name', type=str, required=True)
    parser.add_argument('-robot_ip', type=str)
    parser.add_argument('-sim', type=str, required=True)
    # parser.add_argument('-run_vicon', type=str, required=True)
    parser.add_argument('-x', type=float)
    parser.add_argument('-y', type=float)
    parser.add_argument('-z', type=float)
    parser.add_argument('-Y', type=float, default=0.0)

    args, unknown = parser.parse_known_args()
    if args.sim == '1':
        simulation_on = True
    else:
        simulation_on = False

    if simulation_on:
        # Positional arguments
        sim = 1
        x_init = args.x
        y_init = args.y
        z_init = args.z 
        yaw_init = args.Y
        robot_ip = None
    else:
        sim = 0
        x_init = None
        y_init = None
        z_init = None
        yaw_init = None
        robot_ip = args.robot_ip


    print("Starting", args.robot_name)
    tb = Turtlebot(args.robot_name, robot_ip, sim, x_init,y_init,z_init, yaw_init, simulation_on)

    time.sleep(10)

    try:
        while not rospy.is_shutdown():
            tb.run()
        #If we press ctrl + C, the node will stop.
        rospy.spin()
    except rospy.ROSInterruptException:
        pass