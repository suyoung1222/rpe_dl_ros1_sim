#!/usr/bin/env python3
import math

import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError

# Import the relative pose estimation code (your snippet)
from estimate_rel_pose_diffglue import estimate_relative_pose
from geometry_msgs.msg import PoseStamped, Twist

# ROS messages
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_matrix

################################################################################
# ROS Topics
################################################################################
leader_img_topic = "/Husky1_colorImg"
follower_img_topic = "/Jackal_2_colorImg"
follower_depth_topic = "/Jackal_2_depthImg"
follower_cmd_topic = "/Jackal_2_cmd_vel"
follower_odometry_topic = "/Jackal_2_SelfOdom"


###############################################################################
# Global variables
###############################################################################
bridge = CvBridge()

# Images and depth
latest_leader_img = None  # Grayscale leader image (np.ndarray)
latest_follower_img = None  # Grayscale follower image (np.ndarray)
latest_follower_depth = (
    None  # Depth image from follower, (np.ndarray, in meters)
)

# Desired relative pose (translation, rotation)
desired_translation = np.array([0.0, 0.0, 0.0])  # from YAML
desired_quat = np.array([0.0, 0.0, 0.0, 1.0])  # from YAML

# PID control gains
Kp_x = 0.3  # P-gain for x,y translation error
Kp_yaw = 0.3  # P-gain for yaw error
Ki_x = 0.0  # I-gain for x,y translation error
Ki_yaw = 0.0  # I-gain for yaw error
Kd_x = 0.0  # D-gain for x,y translation error
Kd_yaw = 0.0  # D-gain for yaw error

# PID state
pid_state = {
    "prev_error_x": 0.0,  # Previous error for x
    "prev_error_yaw": 0.0,  # Previous error for yaw
    "integral_x": 0.0,  # Integral of error for x
    "integral_yaw": 0.0,  # Integral of error for yaw
}
# PID limits
pid_limits = {
    "linear": {
        "min": -0.2,  # m/s
        "max": 0.2,  # m/s
    },
    "angular": {
        "min": -0.1,  # rad/s
        "max": 0.1,  # rad/s
    },
}
# PID time step
pid_dt = 0.1  # seconds


def pid_control(
    target_vx,
    target_vyaw,
    current_vx,
    current_vyaw,
    dt=pid_dt,
):
    """
    PID control for velocity command.
    """
    global pid_state

    # Compute errors
    error_x = target_vx - current_vx
    error_yaw = target_vyaw - current_vyaw

    # Proportional term
    p_term_x = Kp_x * error_x
    p_term_yaw = Kp_yaw * error_yaw

    # Integral term
    pid_state["integral_x"] += error_x * dt
    pid_state["integral_yaw"] += error_yaw * dt
    i_term_x = Ki_x * pid_state["integral_x"]
    i_term_yaw = Ki_yaw * pid_state["integral_yaw"]

    # Derivative term
    d_term_x = Kd_x * (error_x - pid_state["prev_error_x"]) / dt
    d_term_yaw = Kd_yaw * (error_yaw - pid_state["prev_error_yaw"]) / dt

    # Update previous errors
    pid_state["prev_error_x"] = error_x
    pid_state["prev_error_yaw"] = error_yaw

    # Compute control command
    cmd_linear_x = p_term_x + i_term_x + d_term_x
    cmd_angular_z = p_term_yaw + i_term_yaw + d_term_yaw

    # Apply limits
    cmd_linear_x = np.clip(
        cmd_linear_x, pid_limits["linear"]["min"], pid_limits["linear"]["max"]
    )
    cmd_angular_z = np.clip(
        cmd_angular_z,
        pid_limits["angular"]["min"],
        pid_limits["angular"]["max"],
    )

    return cmd_linear_x, cmd_angular_z


# #############################################################################
# ROS Publishers
# #############################################################################
cmd_pub = rospy.Publisher(follower_cmd_topic, Twist, queue_size=1)


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


def follower_image_cb(msg):
    """
    Follower camera image callback (grayscale).
    """
    global latest_follower_img
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        latest_follower_img = cv_img
    except CvBridgeError as e:
        rospy.logerr("Follower image conversion failed: %s", str(e))


def follower_depth_cb(msg):
    global latest_follower_depth
    try:
        # Bridge to a 16UC1 image in Python
        depth_img_mm = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

        # Convert the uint16 (mm) image to float32 (m)
        depth_img_m = depth_img_mm.astype(np.float32) / 1000.0

        latest_follower_depth = depth_img_m
    except CvBridgeError as e:
        rospy.logerr("Follower depth conversion failed: %s", str(e))


def folllower_odometry_cb(msg):
    """
    Populate the follower's current velocity.
    """
    global follower_current_vx, follower_current_vyaw
    follower_current_vx = msg.twist.linear.x
    follower_current_vyaw = msg.twist.angular.z


def load_desired_pose(config_path):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    t = np.array(
        data["desired_relative_pose"]["translation"], dtype=np.float32
    )
    q = np.array(data["desired_relative_pose"]["rotation"], dtype=np.float32)
    return t, q


def publish_velocity_cmd_pid(target_vx, target_vyaw):
    """
    Implement a PID controller to compute the velocity command.
    """
    global follower_current_vx, follower_current_vyaw

    # Compute the PID control command
    cmd_linear_x, cmd_angular_z = pid_control(
        target_vx,
        target_vyaw,
        follower_current_vx,
        follower_current_vyaw,
    )

    # Create a Twist message
    cmd_msg = Twist()
    cmd_msg.linear.x = cmd_linear_x
    cmd_msg.angular.z = cmd_angular_z

    # Publish the command
    cmd_pub.publish(cmd_msg)

    print(
        f"Follower cmd: linear_x={cmd_linear_x:.2f}, angular_z={cmd_angular_z:.2f}"
    )


def run_follower_task():
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
            latest_leader_img is not None
            and latest_follower_img is not None
            and latest_follower_depth is not None
        ):

            # Attempt relative pose estimation
            try:
                R, t = estimate_relative_pose(
                    follower_img=latest_follower_img,
                    leader_img=latest_leader_img,
                    follower_depth=latest_follower_depth,
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
                    t_e - desired_translation
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
                    f"Follower position error: {pos_error[0]:2f}, {pos_error[1]:2f}"
                )

                publish_velocity_cmd_pid(0.3 * pos_error[0], 0.3 * yaw_error)

                print(f"Follwer yaw w.r.t leader: {yaw_deg:2f}")
                # print(T_mat)

            except Exception as e:
                # rospy.logwarn(
                #     "Relative pose estimation or control failed: %s", str(e)
                # )
                pass
                # print(e)
                raise e

        rate.sleep()


def main():
    rospy.init_node("run_follower_task", anonymous=True)

    # Load config for desired pose
    config_path = rospy.get_param("~config_path", "./configs/scenario1.yaml")
    global desired_translation, desired_quat
    desired_translation, desired_quat = load_desired_pose(config_path)

    # Subscribers
    rospy.Subscriber(leader_img_topic, Image, leader_image_cb, queue_size=1)
    rospy.Subscriber(
        follower_img_topic, Image, follower_image_cb, queue_size=1
    )
    rospy.Subscriber(
        follower_depth_topic, Image, follower_depth_cb, queue_size=1
    )
    rospy.Subscriber(
        follower_odometry_topic,
        PoseStamped,
        folllower_odometry_cb,
        queue_size=1,
    )

    # Start main loop
    run_follower_task()


if __name__ == "__main__":
    main()
