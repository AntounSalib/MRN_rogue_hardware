#!/usr/bin/env python

import numpy as np
import rospy
import json
import time
import math
import copy
import signal
import os
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import String
from nod_controller import NodController
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion
from data_saver import RobotDataSaver, ImportsDataSaver
from neighbors import sensed_neighbors
from constants import ROBOT_NAMES, EPS, NodConfig, ROGUE_AGENTS, ROGUE_SPEEDS, ORCA_AGENTS, ORCA_DD_AGENTS, MPC_CBF_AGENTS, HUMAN_NAMES, D_SAFE, RESET_TO_START, START_POSITIONS, ACTIVE_ROBOTS
from orca_dd_controller import NHORCAController
from mpc_cbf_controller import MPCCBFController
import rvo2

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
        self.goal_heading = None  # set on first ORCA step
        self.heading_error_integral = 0.0
        self.reset_position_reached = False
        self.info = {
            "name": self.robot_name,
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "heading": 0.0,
        }
        self.target_speed = 0.0

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
            
            # Subscribe to humans
            self.human_states = {}
            for human_name in HUMAN_NAMES:
                self.human_states[human_name] = {'prev_time': None, 'prev_pos': None, 'velocity': [0.0, 0.0]}
                vicon_topic = f'/vicon/{human_name}/{human_name}'
                rospy.Subscriber(vicon_topic, TransformStamped, self.human_callback, callback_args=human_name)
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
        self.nhorca_controller = NHORCAController() if self.robot_name in ORCA_DD_AGENTS else None
        self.mpc_cbf_controller = MPCCBFController() if self.robot_name in MPC_CBF_AGENTS else None
        self.data_saver = RobotDataSaver(self.robot_name)

        # Save configuration files
        config_saver = ImportsDataSaver()
        config_saver.save_all_config_info()

        rospy.on_shutdown(self.tb_stop)

    def tb_stop(self):
        vel_msg = Twist()

        vel_msg.linear.x = 0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0

        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0

        # Publish multiple times to ensure it is received.
        # Guard against the publisher already being closed (e.g. on ROS shutdown).
        for _ in range(3):
            try:
                self.vel_pub.publish(vel_msg)
            except Exception:
                break
            time.sleep(0.1)

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
                alpha = 0.75  # smoothing factor
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

        # # computing linear velocities based on commanded velocity and heading
        # vel_mag = math.hypot(self.info["velocity"][0], self.info["velocity"][1]) + EPS
        # self.info["velocity"][0] = vel_mag * math.cos(self.info["heading"])
        # self.info["velocity"][1] = vel_mag * math.sin(self.info["heading"])

        # --- Publish info as JSON string ---
        info_msg = String(data=json.dumps(self.info))
        self.info_pub.publish(info_msg)

        # print(f"{self.robot_name} pos: {self.info['position']}, vel: {self.info['velocity']}, heading: {self.info['heading']:.2f}")

    def move(self, lin_vel, ang_vel):
        vel_msg = Twist()
        vel_msg.linear.x = lin_vel
        vel_msg.angular.z = ang_vel
        try:
            self.vel_pub.publish(vel_msg)
        except Exception:
            return
        self.commanded_velocity = lin_vel
        # print(f'published: {vel_msg}')
        
    def neighbor_callback(self, msg):
        """
        Update the dictionary of neighbors with latest info
        """
        msg_dict = json.loads(msg.data)
        self.neighbors[msg_dict["name"]] = msg_dict
        # print(f"Received info from {msg_dict['name']}: pos={msg_dict['position']}, vel={msg_dict['velocity']}, heading={msg_dict['heading']}")
 
    def human_callback(self, data, human_name):
        x = data.transform.translation.x
        y = data.transform.translation.y
        current_time = data.header.stamp.to_sec()

        # Orientation
        orient_quat = data.transform.rotation
        orient = [orient_quat.x, orient_quat.y, orient_quat.z, orient_quat.w]
        (_, _, yaw) = euler_from_quaternion(orient)

        # Velocity calculation
        state = self.human_states[human_name]
        vx, vy = state['velocity']

        if state['prev_time'] is not None:
            dt = current_time - state['prev_time']
            if dt > 0:
                vx = (x - state['prev_pos'][0]) / dt
                vy = (y - state['prev_pos'][1]) / dt
        
        state['prev_time'] = current_time
        state['prev_pos'] = [x, y]
        state['velocity'] = [vx, vy]

        # Update neighbors dict
        self.neighbors[human_name] = {"name": human_name, "position": [x, y], "velocity": [vx, vy], "heading": yaw}

    def _init_goal_heading(self):
        x, y = self.info['position']
        if abs(x) >= abs(y):
            self.goal_heading = 0.0 if x < 0 else math.pi
        else:
            self.goal_heading = -math.pi / 2 if y > 0 else math.pi / 2

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

    def _compute_orca_velocity(self):
        if self.goal_heading is None:
            self._init_goal_heading()

        sim = rvo2.PyRVOSimulator(
            0.1,                               # time step
            NodConfig.neighbors.SENSING_RANGE, # neighbor distance
            10,                                # max neighbors
            2.0,                               # time horizon
            2.0,                               # time horizon obstacles
            D_SAFE / 2.0,                      # agent radius
            NodConfig.kin.V_MAX,               # max speed
        )

        ego_id = sim.addAgent(tuple(self.info['position']))
        sim.setAgentVelocity(ego_id, tuple(self.info['velocity']))

        for neighbor_info in self.neighbors.values():
            n_id = sim.addAgent(tuple(neighbor_info['position']))
            sim.setAgentVelocity(n_id, tuple(neighbor_info['velocity']))
            h = neighbor_info['heading']
            sim.setAgentPrefVelocity(n_id, (NodConfig.kin.V_NOMINAL * math.cos(h),
                                            NodConfig.kin.V_NOMINAL * math.sin(h)))

        sim.setAgentPrefVelocity(ego_id, (NodConfig.kin.V_NOMINAL * math.cos(self.goal_heading),
                                          NodConfig.kin.V_NOMINAL * math.sin(self.goal_heading)))
        sim.doStep()
        return sim.getAgentVelocity(ego_id)

    def _run_reset(self):
        if self.robot_name not in START_POSITIONS:
            self.move(0, 0)
            self.rate.sleep()
            return

        goal_x, goal_y, goal_heading = START_POSITIONS[self.robot_name]
        goal = np.array([goal_x, goal_y])
        pos  = np.array(self.info['position'])
        diff = goal - pos
        dist = float(np.linalg.norm(diff))

        if dist < 0.15:
            self.reset_position_reached = True

        if self.reset_position_reached:
            # Position reached — spin to target heading
            heading_error = math.atan2(math.sin(goal_heading - self.info['heading']),
                                       math.cos(goal_heading - self.info['heading']))
            if abs(heading_error) < 0.02:
                self.move(0, 0)
            else:
                ang_vel = math.copysign(
                    min(abs(NodConfig.kin.KAPPA_ANG * heading_error), NodConfig.mpc_cbf.OMEGA_MAX),
                    heading_error)
                self.move(0, ang_vel)
            self.rate.sleep()
            return

        # Use ORCA with goal as preferred velocity direction
        import rvo2
        sim = rvo2.PyRVOSimulator(
            0.1,
            NodConfig.neighbors.SENSING_RANGE,
            10, 2.0, 2.0,
            D_SAFE / 2.0,
            NodConfig.kin.V_MAX,
        )
        ego_id = sim.addAgent(tuple(pos))
        sim.setAgentVelocity(ego_id, tuple(self.info['velocity']))
        for neighbor_info in self.neighbors.values():
            nb_pos = tuple(neighbor_info['position'])
            nb_vel = tuple(neighbor_info['velocity'])
            n_id = sim.addAgent(nb_pos)
            sim.setAgentVelocity(n_id, nb_vel)
            # give neighbors their current velocity as preferred so ORCA predicts their motion
            sim.setAgentPrefVelocity(n_id, nb_vel)

        pref_speed = min(NodConfig.kin.V_NOMINAL, dist * 2.0)
        pref_vel   = (diff / dist) * pref_speed
        sim.setAgentPrefVelocity(ego_id, tuple(pref_vel))
        sim.doStep()
        orca_vel = sim.getAgentVelocity(ego_id)

        v_lin = float(np.clip(math.hypot(orca_vel[0], orca_vel[1]), 0.0, NodConfig.kin.V_MAX))
        # steer toward ORCA output direction, not raw goal direction
        if v_lin > EPS:
            desired_heading = math.atan2(orca_vel[1], orca_vel[0])
        else:
            desired_heading = math.atan2(diff[1], diff[0])
        heading = self.info['heading']
        heading_error = math.atan2(math.sin(desired_heading - heading),
                                   math.cos(desired_heading - heading))
        ang_vel = math.copysign(
            min(abs(NodConfig.kin.KAPPA_ANG * heading_error), NodConfig.mpc_cbf.OMEGA_MAX),
            heading_error)
        self.move(v_lin, ang_vel)
        self.rate.sleep()

    def run(self):
        if RESET_TO_START:
            self._run_reset()
            return



        ego_pos = self.info['position']
        if (abs(ego_pos[0]) > 2.85 or ego_pos[1] > 3.2 or ego_pos[1] < -1.8):
            # print(f"{self.robot_name} BOUNDARY STOP at pos={[round(v,3) for v in ego_pos]}")
            self.move(0, 0)
            return

        sens_neighbors = sensed_neighbors(self.info, self.neighbors)

        if self.robot_name in ROGUE_AGENTS:
            if self.goal_heading is None:
                self._init_goal_heading()
            self.target_speed = ROGUE_SPEEDS.get(self.robot_name, NodConfig.kin.V_ROGUE)
            self.data_saver.save_data(self.info, self.neighbors, sens_neighbors, self.nod_controller, self.target_speed)
            heading = self.info['heading']
            heading_error = math.atan2(math.sin(self.goal_heading - heading),
                                       math.cos(self.goal_heading - heading))
            self.heading_error_integral += heading_error * 0.1
            self.heading_error_integral = math.copysign(
                min(abs(self.heading_error_integral), NodConfig.mpc_cbf.OMEGA_MAX / NodConfig.kin.KAPPA_ANG_I),
                self.heading_error_integral)
            ang_vel = math.copysign(
                min(abs(NodConfig.kin.KAPPA_ANG * heading_error
                        + NodConfig.kin.KAPPA_ANG_I * self.heading_error_integral),
                    NodConfig.mpc_cbf.OMEGA_MAX),
                heading_error + NodConfig.kin.KAPPA_ANG_I / NodConfig.kin.KAPPA_ANG * self.heading_error_integral)
            self.move(self.target_speed, ang_vel)
            self.rate.sleep()
            return

        if self.robot_name in ORCA_AGENTS:
            orca_vel = self._compute_orca_velocity()
            v_lin = float(np.clip(math.hypot(orca_vel[0], orca_vel[1]), 0.0, NodConfig.kin.V_MAX))
            if v_lin > EPS:
                desired_heading = math.atan2(orca_vel[1], orca_vel[0])
                heading = self.info['heading']
                heading_error = math.atan2(math.sin(desired_heading - heading),
                                           math.cos(desired_heading - heading))
                ang_vel = NodConfig.kin.KAPPA_ANG * heading_error
            else:
                ang_vel = 0.0
            self.data_saver.save_data(self.info, self.neighbors, sens_neighbors, self.nod_controller, v_lin)
            self.move(v_lin, ang_vel)
            self.rate.sleep()
            return

        if self.robot_name in ORCA_DD_AGENTS:
            if self.goal_heading is None:
                self._init_goal_heading()
            v_lin, ang_vel = self.nhorca_controller.compute_velocity(self.info, self.neighbors, self.goal_heading)
            self.data_saver.save_data(self.info, self.neighbors, sens_neighbors, self.nod_controller, v_lin)
            self.move(v_lin, ang_vel)
            self.rate.sleep()
            return

        if self.robot_name in MPC_CBF_AGENTS:
            if self.goal_heading is None:
                self._init_goal_heading()
            v_lin, ang_vel = self.mpc_cbf_controller.compute_velocity(self.info, self.neighbors, self.goal_heading)
            self.data_saver.save_data(self.info, self.neighbors, sens_neighbors, self.nod_controller, v_lin)
            self.move(v_lin, ang_vel)
            self.rate.sleep()
            return

        if self.goal_heading is None:
            self._init_goal_heading()

        self.target_speed = self.nod_controller.update_opinion(self.info, self.neighbors, time.time())
        self.data_saver.save_data(self.info, self.neighbors, sens_neighbors, self.nod_controller, self.target_speed)
        # print(f"{self.robot_name} target speed: {self.target_speed:.3f}")
        v_command = self._get_v_commanded(self.target_speed)

        heading = self.info['heading']
        heading_error = math.atan2(math.sin(self.goal_heading - heading),
                                   math.cos(self.goal_heading - heading))
        self.heading_error_integral += heading_error * 0.1  # dt = 0.1s
        self.heading_error_integral = math.copysign(     # anti-windup clamp
            min(abs(self.heading_error_integral), NodConfig.mpc_cbf.OMEGA_MAX / NodConfig.kin.KAPPA_ANG_I),
            self.heading_error_integral)
        ang_vel = math.copysign(
            min(abs(NodConfig.kin.KAPPA_ANG * heading_error
                    + NodConfig.kin.KAPPA_ANG_I * self.heading_error_integral),
                NodConfig.mpc_cbf.OMEGA_MAX),
            heading_error + NodConfig.kin.KAPPA_ANG_I / NodConfig.kin.KAPPA_ANG * self.heading_error_integral)

        self.move(v_command, ang_vel)

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

    def handle_sigtstp(signum, frame):
        tb.tb_stop()
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTSTP)
    signal.signal(signal.SIGTSTP, handle_sigtstp)

    time.sleep(3)

    # Synchronize start via ROS parameter server (centralized, no pub/sub timing issues)
    rospy.set_param(f'/ready/{tb.robot_name}', True)
    rospy.loginfo(f"{tb.robot_name} ready, waiting for: {ACTIVE_ROBOTS - {tb.robot_name}}")

    while not rospy.is_shutdown():
        if all(rospy.get_param(f'/ready/{r}', False) for r in ACTIVE_ROBOTS):
            break
        time.sleep(0.1)

    # First robot to clear barrier sets the common start time
    if not rospy.has_param('/start_time'):
        rospy.set_param('/start_time', time.time())
    start_time = rospy.get_param('/start_time')
    rospy.loginfo(f"{tb.robot_name} all robots ready, starting")

    if tb.robot_name in ROGUE_AGENTS:
        rogue_delay = 0
        wake_time = start_time + rogue_delay
        sleep_remaining = wake_time - time.time()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)

    try:
        while not rospy.is_shutdown():
            tb.run()
        #If we press ctrl + C, the node will stop.
    except rospy.ROSInterruptException:
        pass
    finally:
        tb.tb_stop()