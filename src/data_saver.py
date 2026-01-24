#!/usr/bin/env python3

from constants import TRIAL_ID, TRIAL_SEED, NodConfig, ROBOT_NAMES

# ROS Libraries/Packages
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool, Float32, Int16
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, TransformStamped
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SpawnModel, SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState

import os
import csv
import shutil
import numpy as np


# ===========================================================
# ==============  PATH AND FOLDER MANAGEMENT  ===============
# ===========================================================

BASE_PATH = os.path.expanduser("~/catkin_ws/src/MRN_rogue_hardware/data")

TRIAL_PATH = os.path.join(BASE_PATH, f"ID{TRIAL_ID}", f"trial_{TRIAL_SEED}")
os.makedirs(TRIAL_PATH, exist_ok=True)


# ===========================================================
# ==================   CONSTANTS SAVER   ======================
# ===========================================================

class ImportsDataSaver:
    """Saves a copy of constants.py for record keeping."""

    def __init__(self):
        self.filepath = TRIAL_PATH

    def save_all_config_info(self):
        # Copy current constants.py
        src_path = os.path.join(os.path.dirname(__file__), "constants.py")
        dst_path = os.path.join(self.filepath, "constants_copy.py")
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            print("SAVED CONSTANTS FILE")
            shutil.copy(src_path, dst_path)

# ===========================================================
# ==================   ROBOT DATA SAVER   ===================
# ===========================================================

class RobotDataSaver:
    def __init__(self, name):
        self.name = name
        self.folder = os.path.join(TRIAL_PATH, self.name)
        os.makedirs(self.folder, exist_ok=True)

        # Filename
        self.file_path = os.path.join(self.folder, f"{self.name}_data.csv")
        
        # Identify other robots for column headers
        self.other_robots = [r for r in ROBOT_NAMES if r != self.name]

        self._init_files()

    def _init_files(self):
        """Create headers for file if not present."""
        if not os.path.exists(self.file_path):
            headers = ["t", "x", "y", "heading", "opinion", "attention", "target_speed"]
            for r in self.other_robots:
                headers.extend([
                    f"p_att_{r}",
                    f"p_coop_{r}",
                    f"p_coop_att_{r}",
                    f"vx_{r}",
                    f"vy_{r}"
                ])
            
            with open(self.file_path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def save_data(self, ego_info, neighbors, sensed_neighbors, nod_controller, target_speed):

        t = rospy.get_time()
        
        # Robot Info
        x, y = ego_info["position"]
        heading = ego_info["heading"]
        
        # Nod Controller Info
        z = nod_controller.z
        u = nod_controller.u

        row = [t, x, y, heading, z, u, target_speed]

        # Pairwise Info
        for r in self.other_robots:
            if r in sensed_neighbors:
                # Pairwise U
                p_att = nod_controller.pairwise_u[r] if r in nod_controller.pairwise_u else 0.0
                # Pairwise Cooperation
                p_coop = nod_controller.pairwise_cooperation[r] if r in nod_controller.pairwise_cooperation else 0.0
                # Pairwise Cooperation Attention
                p_coop_att = nod_controller.pairwise_cooperation_attention[r] if r in nod_controller.pairwise_cooperation_attention else 0.0
                
                # Neighbor Speed
                nb_data = neighbors[r]
                vx, vy = nb_data['velocity']
                row.extend([p_att, p_coop, p_coop_att, vx, vy])
            else:
                row.extend([np.nan, np.nan, np.nan, np.nan, np.nan])

        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        
        
def main():
    rospy.init_node("data_saver")
    try:
        Idt = ImportsDataSaver()
        Idt.save_all_config_info()
        rospy.loginfo(f"Data folder ready: {TRIAL_PATH}")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()