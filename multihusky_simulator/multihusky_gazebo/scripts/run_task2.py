#!/usr/bin/env python3
import math
import threading

import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from estimate_rel_pose_diffglue import estimate_relative_pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pid import pid_control
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_matrix

################################################################################
# ROS Topics
################################################################################
leader_img_topic = "/UnitreeB1_colorImg"
follower1_img_topic = "/Husky1_colorImg"
follower1_depth_topic = "/Husky1_depthImg"
follower1_cmd_topic = "Husky1_cmd_vel"
follower1_odometry_topic = "/Husky1_selfodom"
follower2_img_topic = "/Jackal_2_colorImg"
follower2_depth_topic = "/Jackal_2_depthImg"
follower2_cmd_topic = "/Jackal_2_cmd_vel"
follower2_odometry_topic = "/Jackal_2_selfodom"


###############################################################################
# Global variables
###############################################################################
bridge = CvBridge()

# Images and depth
latest_leader_img = None
# For Follower1
latest_follower1_img = None
latest_follower1_depth = None
follower1_current_vx = 0.0
follower1_current_vyaw = 0.0

# For Follower2
latest_follower2_img = None
latest_follower2_depth = None
follower2_current_vx = 0.0
follower2_current_vyaw = 0.0

# Desired relative pose (translation, rotation)
desired_translation = np.array([0.0, 0.0, 0.0])  # from YAML
desired_quat = np.array([0.0, 0.0, 0.0, 1.0])  # from YAML

# #############################################################################
# ROS Publishers
# #############################################################################
# cmd_pub = rospy.Publisher(follower_cmd_topic, Twist, queue_size=1)
follower1_cmd_pub = rospy.Publisher(follower1_cmd_topic, Twist, queue_size=1)
follower2_cmd_pub = rospy.Publisher(follower2_cmd_topic, Twist, queue_size=1)


###############################################################################
# Callbacks
###############################################################################
def leader_image_cb(msg):
    """
    Leader camera image callback (grayscale).
    """
    global latest_leader_img
    try:
        # Convert to grayscale np.ndarray
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        latest_leader_img = cv_img
    except CvBridgeError as e:
        rospy.logerr("Leader image conversion failed: %s", str(e))


def follower1_image_cb(msg):
    """
    Follower camera image callback (grayscale).
    """
    global latest_follower1_img
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        latest_follower1_img = cv_img
    except CvBridgeError as e:
        rospy.logerr("Follower image conversion failed: %s", str(e))


def follower2_image_cb(msg):
    """
    Follower camera image callback (grayscale).
    """
    global latest_follower2_img
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        latest_follower2_img = cv_img
    except CvBridgeError as e:
        rospy.logerr("Follower image conversion failed: %s", str(e))


def follower1_depth_cb(msg):
    global latest_follower1_depth
    try:
        # Bridge to a 16UC1 image in Python
        depth_img_mm = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

        # Convert the uint16 (mm) image to float32 (m)
        depth_img_m = depth_img_mm.astype(np.float32) / 1000.0

        latest_follower1_depth = depth_img_m
    except CvBridgeError as e:
        rospy.logerr("Follower depth conversion failed: %s", str(e))


def follower2_depth_cb(msg):
    global latest_follower2_depth
    try:
        # Bridge to a 16UC1 image in Python
        depth_img_mm = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

        # Convert the uint16 (mm) image to float32 (m)
        depth_img_m = depth_img_mm.astype(np.float32) / 1000.0

        latest_follower2_depth = depth_img_m
    except CvBridgeError as e:
        rospy.logerr("Follower depth conversion failed: %s", str(e))


def follower1_odometry_cb(msg):
    """
    Populate the follower's current velocity.
    """
    global follower1_current_vx, follower1_current_vyaw
    follower1_current_vx = msg.twist.twist.linear.x
    follower1_current_vyaw = msg.twist.twist.angular.z


def follower2_odometry_cb(msg):
    """
    Populate the follower's current velocity.
    """
    global follower2_current_vx, follower2_current_vyaw
    follower2_current_vx = msg.twist.twist.linear.x
    follower2_current_vyaw = msg.twist.twist.angular.z


def load_desired_pose(config_path):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    t1 = np.array(
        data["desired_relative_pose1"]["translation"], dtype=np.float32
    )
    q1 = np.array(data["desired_relative_pose1"]["rotation"], dtype=np.float32)
    t2 = np.array(
        data["desired_relative_pose2"]["translation"], dtype=np.float32
    )
    q2 = np.array(data["desired_relative_pose2"]["rotation"], dtype=np.float32)
    return t1, q1, t2, q2


def publish_velocity_cmd_pid1(target_vx, target_vyaw):
    """
    Implement a PID controller to compute the velocity command.
    """
    global follower1_current_vx, follower1_current_vyaw

    # Compute the PID control command
    cmd_linear_x, cmd_angular_z = pid_control(
        target_vx,
        target_vyaw,
        follower1_current_vx,
        follower1_current_vyaw,
    )

    # Create a Twist message
    cmd_msg = Twist()
    cmd_msg.linear.x = cmd_linear_x
    cmd_msg.angular.z = cmd_angular_z

    # Publish the command
    follower1_cmd_pub.publish(cmd_msg)

    print(
        f"Follower1 cmd: linear_x={cmd_linear_x:.2f}, angular_z={cmd_angular_z:.2f}"
    )


def publish_velocity_cmd_pid2(target_vx, target_vyaw):
    """
    Implement a PID controller to compute the velocity command.
    """
    global follower2_current_vx, follower2_current_vyaw

    # Compute the PID control command
    cmd_linear_x, cmd_angular_z = pid_control(
        target_vx,
        target_vyaw,
        follower2_current_vx,
        follower2_current_vyaw,
    )

    # Create a Twist message
    cmd_msg = Twist()
    cmd_msg.linear.x = cmd_linear_x
    cmd_msg.angular.z = cmd_angular_z

    # Publish the command
    follower2_cmd_pub.publish(cmd_msg)

    print(
        f"Follower2 cmd: linear_x={cmd_linear_x:.2f}, angular_z={cmd_angular_z:.2f}"
    )


def run_follower_task1():
    """
    Main loop to:
      - wait for leader/follower images + follower depth
      - estimate the relative pose
      - compare to desired pose
      - publish velocity commands to move follower
    """
    # pose_pub = rospy.Publisher(
    #     "/relative_pose/follower_to_leader", PoseStamped, queue_size=1
    # )

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if (
            latest_leader_img is not None
            and latest_follower1_img is not None
            and latest_follower1_depth is not None
        ):

            # Attempt relative pose estimation
            try:
                R, t = estimate_relative_pose(
                    follower_img=latest_follower1_img,
                    leader_img=latest_leader_img,
                    follower_depth=latest_follower1_depth,
                    # If you have camera intrinsics, pass them here.
                )
                # Publish the pose
                # pose_msg = PoseStamped()
                # pose_msg.header.stamp = rospy.Time.now()
                # pose_msg.header.frame_id = "leader_base"

                # Build 4x4 to get a quaternion
                T_mat = np.eye(4)
                T_mat[:3, :3] = R
                T_mat[:3, 3] = t.ravel()
                # print("rel pose of cam frame")
                # print(T_mat)
                T_conv = np.array(
                    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
                )
                # Transform leader body frame to follower body frame
                T_mat = T_conv.T @ T_mat @ T_conv
                yaw_rad = np.arctan2(T_mat[1][0], T_mat[0][0])
                yaw_deg = np.degrees(yaw_rad)
                print("rel pose of")
                print(T_mat)

                # Convert to quaternionS
                q = quaternion_from_matrix(T_mat)
                # pose_msg.pose.position.x = T_mat[0][3]
                # pose_msg.pose.position.y = T_mat[1][3]
                # pose_msg.pose.position.z = T_mat[2][3]
                # pose_msg.pose.orientation.x = q[0]
                # pose_msg.pose.orientation.y = q[1]
                # pose_msg.pose.orientation.z = q[2]
                # pose_msg.pose.orientation.w = q[3]
                # pose_pub.publish(pose_msg)

                # Compare with desired translation
                t_e = T_mat[:3, 3]  # Extract translation from T_ma
                pos_error = (
                    t_e - desired_translation1
                )  # simple vector difference
                desired_yaw = 0
                yaw_error = yaw_rad - desired_yaw

                pos_tolerance = 0.2
                yaw_tolerance = 0.1
                if np.linalg.norm(pos_error) < pos_tolerance:
                    pos_error = np.zeros(3)

                    if np.abs(yaw_error) < yaw_tolerance:
                        yaw_error = 0

                # We'll do a naive 2D controller, ignoring Z and rotation for demonstration
                print(
                    f"Follower1 position error: {pos_error[0]:2f}, {pos_error[1]:2f}"
                )

                publish_velocity_cmd_pid1(0.3 * pos_error[0], 0.3 * yaw_error)

                print(f"Follwer1 yaw w.r.t leader: {yaw_deg:2f}")
                # print(T_mat)

            except Exception as e:
                # rospy.logwarn(
                #     "Relative pose estimation or control failed: %s", str(e)
                # )
                pass
                # print(e)
                # raise e

        rate.sleep()


def run_follower_task2():
    """
    Main loop to:
      - wait for leader/follower images + follower depth
      - estimate the relative pose
      - compare to desired pose
      - publish velocity commands to move follower
    """
    # pose_pub = rospy.Publisher(
    #     "/relative_pose/follower_to_leader", PoseStamped, queue_size=1
    # )

    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():
        if (
            latest_follower1_img is not None
            and latest_follower2_img is not None
            and latest_follower2_depth is not None
        ):

            # Attempt relative pose estimation
            try:
                R, t = estimate_relative_pose(
                    follower_img=latest_follower2_img,
                    leader_img=latest_follower1_img,
                    follower_depth=latest_follower2_depth,
                    # If you have camera intrinsics, pass them here.
                )
                # Publish the pose
                # pose_msg = PoseStamped()
                # pose_msg.header.stamp = rospy.Time.now()
                # pose_msg.header.frame_id = "leader_base"

                # Build 4x4 to get a quaternion
                T_mat = np.eye(4)
                T_mat[:3, :3] = R
                T_mat[:3, 3] = t.ravel()
                # print("rel pose of cam frame")
                # print(T_mat)
                T_conv = np.array(
                    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
                )
                # Transform leader body frame to follower body frame
                T_mat = T_conv.T @ T_mat @ T_conv
                yaw_rad = np.arctan2(T_mat[1][0], T_mat[0][0])
                yaw_deg = np.degrees(yaw_rad)
                print("rel pose of")
                print(T_mat)

                # Convert to quaternionS
                q = quaternion_from_matrix(T_mat)
                # pose_msg.pose.position.x = T_mat[0][3]
                # pose_msg.pose.position.y = T_mat[1][3]
                # pose_msg.pose.position.z = T_mat[2][3]
                # pose_msg.pose.orientation.x = q[0]
                # pose_msg.pose.orientation.y = q[1]
                # pose_msg.pose.orientation.z = q[2]
                # pose_msg.pose.orientation.w = q[3]
                # pose_pub.publish(pose_msg)

                # Compare with desired translation
                t_e = T_mat[:3, 3]  # Extract translation from T_ma
                pos_error = (
                    t_e - desired_translation2
                )  # simple vector difference
                desired_yaw = 0
                yaw_error = yaw_rad - desired_yaw

                pos_tolerance = 0.2
                yaw_tolerance = 0.1
                if np.linalg.norm(pos_error) < pos_tolerance:
                    pos_error = np.zeros(3)

                    if np.abs(yaw_error) < yaw_tolerance:
                        yaw_error = 0

                # We'll do a naive 2D controller, ignoring Z and rotation for demonstration
                print(
                    f"Follower2 position error: {pos_error[0]:2f}, {pos_error[1]:2f}"
                )

                publish_velocity_cmd_pid2(0.3 * pos_error[0], 0.3 * yaw_error)

                print(f"Follwer2 yaw w.r.t leader: {yaw_deg:2f}")
                # print(T_mat)

            except Exception as e:
                # rospy.logwarn(
                #     "Relative pose estimation or control failed: %s", str(e)
                # )
                pass
                # print(e)
                # raise e

        rate.sleep()


def main():
    rospy.init_node("run_follower_task", anonymous=True)

    # Load config for desired pose
    config_path = rospy.get_param("~config_path", "./configs/scenario2.yaml")
    global desired_translation1, desired_quat1, desired_translation2, desired_quat2
    (
        desired_translation1,
        desired_quat1,
        desired_translation2,
        desired_quat2,
    ) = load_desired_pose(config_path)

    # Subscribers
    rospy.Subscriber(leader_img_topic, Image, leader_image_cb, queue_size=1)
    rospy.Subscriber(
        follower1_img_topic, Image, follower1_image_cb, queue_size=1
    )
    rospy.Subscriber(
        follower1_depth_topic, Image, follower1_depth_cb, queue_size=1
    )
    rospy.Subscriber(
        follower1_odometry_topic,
        Odometry,
        follower1_odometry_cb,
        queue_size=1,
    )

    # Subscribers
    rospy.Subscriber(follower1_img_topic, Image, leader_image_cb, queue_size=1)
    rospy.Subscriber(
        follower2_img_topic, Image, follower2_image_cb, queue_size=1
    )
    rospy.Subscriber(
        follower2_depth_topic, Image, follower2_depth_cb, queue_size=1
    )
    rospy.Subscriber(
        follower2_odometry_topic,
        Odometry,
        follower2_odometry_cb,
        queue_size=1,
    )

    # Start main loop
    # run_follower_task()
    # Start both follower threads
    threading.Thread(target=run_follower_task1).start()
    threading.Thread(target=run_follower_task2).start()
    rospy.spin()


if __name__ == "__main__":
    main()
