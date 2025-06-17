#!/usr/bin/env python3

import math
import threading
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_matrix
from multiprocessing import Queue
from estimate_rel_pose_diffglue import estimate_relative_pose
from geometry_msgs.msg import PoseStamped


# PID gains
Kp_x = 0.3
Kp_yaw = 0.3
Ki_x = 0.0
Ki_yaw = 0.0
Kd_x = 0.0
Kd_yaw = 0.0
pid_dt = 0.1

pid_state = {
    "prev_error_x": 0.0,
    "prev_error_yaw": 0.0,
    "integral_x": 0.0,
    "integral_yaw": 0.0,
}

pid_limits = {
    "linear": {"min": -0.5, "max": 0.5},
    "angular": {"min": -0.2, "max": 0.2},
}

def pid_control(target_vx, target_vyaw, current_vx, current_vyaw, dt=pid_dt):
    global pid_state
    error_x = target_vx - current_vx
    error_yaw = target_vyaw - current_vyaw

    pid_state["integral_x"] += error_x * dt
    pid_state["integral_yaw"] += error_yaw * dt

    d_term_x = Kd_x * (error_x - pid_state["prev_error_x"]) / dt
    d_term_yaw = Kd_yaw * (error_yaw - pid_state["prev_error_yaw"]) / dt
    pid_state["prev_error_x"] = error_x
    pid_state["prev_error_yaw"] = error_yaw

    cmd_linear_x = Kp_x * error_x + Ki_x * pid_state["integral_x"] + d_term_x
    cmd_angular_z = Kp_yaw * error_yaw + Ki_yaw * pid_state["integral_yaw"] + d_term_yaw

    cmd_linear_x = -np.clip(cmd_linear_x, pid_limits["linear"]["min"], pid_limits["linear"]["max"])
    cmd_angular_z = -np.clip(cmd_angular_z, pid_limits["angular"]["min"], pid_limits["angular"]["max"])

    return cmd_linear_x, cmd_angular_z

class RobotFollower:
    def __init__(self, name, image_topic, depth_topic, odometry_topic, cmd_topic, desired_translation, desired_yaw, leader_img_getter):
        self.name = name
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.odom_topic = odometry_topic
        self.cmd_topic = cmd_topic
        self.desired_translation = desired_translation
        self.desired_yaw = desired_yaw
        self.get_leader_image = leader_img_getter

        self.bridge = CvBridge()
        self.latest_img = Queue(1)
        self.latest_depth = Queue(1)
        self.current_vx = 0.0
        self.current_vyaw = 0.0

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pose_pub = rospy.Publisher(f"{self.name}/estimated_pose", PoseStamped, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.image_cb)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

    def image_cb(self, msg):
        try:
            if not self.latest_img.full():
                self.latest_img.put_nowait(self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8"))
        except CvBridgeError as e:
            rospy.logerr(f"{self.name} image error: {e}")

    def depth_cb(self, msg):
        try:
            if not self.latest_depth.full():
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                self.latest_depth.put_nowait(depth_img)
        except CvBridgeError as e:
            rospy.logerr(f"{self.name} depth error: {e}")

    def odom_cb(self, msg):
        self.current_vx = msg.twist.twist.linear.x
        self.current_vyaw = msg.twist.twist.angular.z

    def publish_cmd(self):
        cmd_linear_x, cmd_angular_z = pid_control(self.target_vx, self.target_vyaw, self.current_vx, self.current_vyaw)
        cmd_msg = Twist()
        cmd_msg.linear.x = cmd_linear_x
        cmd_msg.angular.z = cmd_angular_z
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo(f"{self.name} cmd: vx={cmd_linear_x:.2f}, vyaw={cmd_angular_z:.2f}")

    def publish_estimated_pose(self, R, t):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "odom"  # Change to "map" or other frame if needed

        pose_msg.pose.position.x = t[0]
        pose_msg.pose.position.y = t[1]
        pose_msg.pose.position.z = t[2]

        # Convert rotation mapose_msgtrix to quaternion
        quat = quaternion_from_matrix(np.vstack([
            np.hstack([R, np.zeros((3, 1))]),
            [0, 0, 0, 1]
        ]))

        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)
        rospy.loginfo(f"=============================Estimated Pose============================={pose_msg}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            leader_img = self.get_leader_image()
            if leader_img is not None and not self.latest_img.empty() and not self.latest_depth.empty():
                try:
                    R, t, pose_estimation_valid = estimate_relative_pose(
                        follower_img=self.latest_img.get_nowait(),
                        leader_img=leader_img,
                        follower_depth=self.latest_depth.get_nowait()
                    )
                    T_mat = np.eye(4)
                    T_mat[:3, :3] = R
                    T_mat[:3, 3] = t.ravel()
                    T_conv = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
                    T_mat = T_conv.T @ T_mat @ T_conv

                    yaw_rad = math.atan2(T_mat[1][0], T_mat[0][0])
                    pos_error = T_mat[:3, 3] - self.desired_translation
                    distance = np.linalg.norm(pos_error)

                    yaw_error = yaw_rad - self.desired_yaw
                    if distance < 0.2:
                        pos_error = np.zeros(3)
                        if abs(yaw_error) < 0.1:
                            yaw_error = 0.0
                    rospy.loginfo(f"pose estimation: {T_mat}")
                    if pose_estimation_valid:
                        self.publish_estimated_pose(T_mat[:3, :3], T_mat[:3, 3])
                    self.target_vx = pos_error[0]
                    self.target_vyaw = yaw_error

                    self.publish_cmd()
                except Exception as e:
                    rospy.logwarn(f"{self.name} failed pose estimation: {e}")
            rate.sleep()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    rospy.init_node("multi_robot_follower")
    config_path = rospy.get_param("~config_path", "./configs/sim_1.yaml")
    config = load_config(config_path)

    bridge = CvBridge()
    latest_images = {}

    def make_img_cb(robot_name):
        def callback(msg):
            try:
                latest_images[robot_name] = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            except CvBridgeError as e:
                rospy.logerr(f"{robot_name} image conversion failed: {e}")
        return callback

    leader_name = ""
    for robot in config["robots"]:
        if robot["role"] == "leader":
            rospy.Subscriber(robot["image_topic"], Image, make_img_cb(robot["name"]))
            leader_name = robot["name"]

    def get_leader_img():
        return latest_images.get(leader_name)

    followers = []
    for robot in config["robots"]:
        if robot["role"] == "follower":
            follower = RobotFollower(
                name=robot["name"],
                image_topic=robot["image_topic"],
                depth_topic=robot["depth_topic"],
                odometry_topic=robot["odometry_topic"],
                cmd_topic=robot["cmd_topic"],
                desired_translation=np.array(robot["desired_translation"], dtype=np.float32),
                desired_yaw=np.array(robot["desired_yaw"]),
                leader_img_getter=get_leader_img,
            )
            followers.append(follower)

    for follower in followers:
        threading.Thread(target=follower.run).start()

    rospy.spin()

if __name__ == "__main__":
    main()
