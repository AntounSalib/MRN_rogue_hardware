#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import numpy as np
import tf.transformations as tft

def create_edge_sdf(length, idx):
    # SDF without <pose> tag, with visual and collision
    sdf = f"""
<sdf version="1.6">
  <model name="boundary_edge_{idx}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{length} 0.05 0.01</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{length} 0.05 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    return sdf

def main():
    rospy.init_node('spawn_polygon_edges')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # Environment layout
    m = 1.1
    corner_x0_y0: np.ndarray        = m*np.array([-3.27, -1.23])
    corner_x0_y1: np.ndarray        = m*np.array([-3.1, 3.32])
    corner_x1_y1: np.ndarray        = m*np.array([2.69, 2.73])
    corner_x1_y0: np.ndarray        = m*np.array([2.83, -2.02])

    # Define your polygon corners here (example rectangle)
    corners = np.array([
            corner_x0_y0,
            corner_x0_y1,
            corner_x1_y1,
            corner_x1_y0
        ])
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        length = np.linalg.norm(p2 - p1)
        sdf = create_edge_sdf(length, i)

        midpoint = (p1 + p2) / 2
        yaw = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        quat = tft.quaternion_from_euler(0, 0, yaw)

        pose = Pose()
        pose.position.x = midpoint[0]
        pose.position.y = midpoint[1]
        pose.position.z = 0.01  # small height so it's visible just above ground
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        try:
            spawn_model(
                model_name=f"boundary_edge_{i}",
                model_xml=sdf,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world"
            )
            rospy.loginfo(f"Spawned edge {i}")
        except Exception as e:
            rospy.logerr(f"Failed to spawn edge {i}: {e}")

    # Wait a bit before finishing so Gazebo can process
    rospy.sleep(2)

if __name__ == "__main__":
    main()
