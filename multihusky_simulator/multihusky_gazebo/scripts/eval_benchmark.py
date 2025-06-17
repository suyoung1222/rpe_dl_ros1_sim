#!/usr/bin/env python3
import rospy
import rosbag
import argparse
import numpy as np
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_matrix
from tf.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import pdb
import matplotlib.pyplot as plt
from estimate_rel_pose_diffglue_eval import estimate_relative_pose
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

def read_image(msg):
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    except Exception as e:
        rospy.logerr(f"Image decode failed: {e}")
        return None

def read_depth(msg):
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
    except Exception as e:
        rospy.logerr(f"Image decode failed: {e}")
        return None


def calculate_ATE(gt_pose, est_pose):
    err =  np.linalg.norm(np.array(gt_pose[:3]) - np.array(est_pose[:3]))
    return err

def calculate_RMD(gt_pose, est_pose):
    R_gt = R.from_quat(gt_pose[3:]).as_matrix()
    R_est = R.from_quat(est_pose[3:]).as_matrix()
    diff_rot = R_gt.T @ R_est
    angle = np.arccos(np.clip((np.trace(diff_rot) - 1) / 2, -1.0, 1.0))
    err = np.degrees(abs(angle))
    return err  # in degrees

def parse_pose(pose_msg):
    pos = pose_msg.position
    ori = pose_msg.orientation
    return [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

def find_closest_gt(gt_data, t):
    closest = min(gt_data, key=lambda x: abs((x[0] - t).to_sec()))
    return closest[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", type=str, help="Path to ROS bag file")
    parser.add_argument("--leader", type=str, default="h1")
    parser.add_argument("--follower", type=str, default="h2")
    parser.add_argument("--gt_topic", type=str, default="/gazebo/model_states")
    args = parser.parse_args()

    leader_image_topic = "/" + args.leader + "/realsense/color/image_raw"
    follower_image_topic = "/" + args.follower + "/realsense/color/image_raw"
    follower_depth_topic = "/" + args.follower + "/realsense/depth/image_rect_raw"

    leader_images = []  
    follower_images = []
    follower_depths = []
    gt_poses = []   # (timestamp, pose [x,y,z,qx,qy,qz,qw]) from Husky_h2 w.r.t. Husky_h1

    with rosbag.Bag(args.bagfile, "r") as bag:
        model_states_msgs = []
        for topic, msg, t in bag.read_messages():
            if topic == leader_image_topic:
                leader_image = read_image(msg)
                if leader_image is not None:
                    leader_images.append((t, leader_image))
            elif topic == follower_image_topic:
                follower_image = read_image(msg)
                if follower_image is not None:
                    follower_images.append((t, follower_image))
            elif topic == follower_depth_topic:
                follower_depth = read_depth(msg)
                if follower_depth is not None:
                    follower_depths.append((t, follower_depth))
            elif topic == args.gt_topic:
                model_states_msgs.append((t, msg))

    if not leader_image_topic:
        print("No data found images")
        return
    if not model_states_msgs:
        print("No data found for ground truth!")
        return

    # Build a dictionary of ground truth over time
    gt_data = []
    for t, msg in model_states_msgs:
        try:
            idx1 = msg.name.index("Husky_h1")
            idx2 = msg.name.index("Husky_h2")
            pose1 = msg.pose[idx1]
            pose2 = msg.pose[idx2]

            # Relative transform T_12 = inv(T1) * T2
            p1 = parse_pose(pose1)
            p2 = parse_pose(pose2)

            T1 = np.eye(4)
            T1[:3, 3] = p1[:3]
            T1[:3, :3] = quaternion_matrix(p1[3:])[:3, :3]

            T2 = np.eye(4)
            T2[:3, 3] = p2[:3]
            T2[:3, :3] = quaternion_matrix(p2[3:])[:3, :3]

            T_rel = np.linalg.inv(T1) @ T2
            pos = T_rel[:3, 3]
            rot = R.from_matrix(T_rel[:3, :3]).as_quat()  # x, y, z, w

            gt_pose = list(pos) + list(rot)
            gt_data.append((t, gt_pose))

        except ValueError:
            continue

    # Estimate Pose
    est_poses = []
    for (t_est, follower_img), (_, depth_img), (_, leader_img) in zip(follower_images, follower_depths, leader_images):
        R_est, t_est_vec, valid = estimate_relative_pose(
            follower_img=follower_img,
            leader_img=leader_img,
            follower_depth=depth_img
        )

        if valid:
            T_mat = np.eye(4)
            T_mat[:3, :3] = R_est
            T_mat[:3, 3] = t_est_vec.ravel()

            T_conv = np.array([
                [0, -1, 0, 0],
                [0,  0, -1, 0],
                [1,  0, 0, 0],
                [0,  0, 0, 1]
            ])
            T_mat = T_conv.T @ T_mat @ T_conv

            pos = T_mat[:3, 3]
            quat = quaternion_from_matrix(T_mat)
        
            est_pose = list(pos) + list(quat)
            est_poses.append((t_est, est_pose))




    # Evaluate
    ates = []
    ates_succ = []
    rmds = []
    rmds_succ = []
    timestamps = []
    timestamps_succ = []

    for t_est, est_pose in est_poses:
        gt_pose = find_closest_gt(gt_data, t_est)

        ate = calculate_ATE(gt_pose, est_pose)
        rmd = calculate_RMD(gt_pose, est_pose)
        ates.append(ate)
        rmds.append(rmd)
        timestamps.append(t_est.to_sec())

        if ate < 100 and rmd < 100:
            ates_succ.append(ate)
            rmds_succ.append(rmd)
            timestamps_succ.append(t_est.to_sec())


    def summarize(name, values):
        values = np.array(values)
        print(f"====== {name} ======")
        print(f"Mean:  {np.mean(values):.4f}")
        print(f"RMSE:  {np.sqrt(np.mean(values**2)):.4f}")
        print(f"Max:   {np.max(values):.4f}")
        print(f"Min:   {np.min(values):.4f}")
        print()

    summarize("ATE (m)", ates)
    summarize("RMD (deg)", rmds)

    summarize("ATE (m)", ates_succ)
    summarize("RMD (deg)", rmds_succ)



    # Plot results as dots
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.scatter(timestamps, ates, label="ATE (m)", color='tab:blue', s=20)
    plt.ylabel("ATE (m)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(timestamps, rmds, label="RMD (deg)", color='tab:orange', s=20)
    plt.xlabel("Time (s)")
    plt.ylabel("RMD (deg)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("accuracy_plot_whole.png")
    print("📊 Saved plot to accuracy_plot.png")
    plt.show()  


    # Plot results as dots
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.scatter(timestamps_succ, ates_succ, label="ATE (m)", color='tab:blue', s=20)
    plt.ylabel("ATE (m)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(timestamps_succ, rmds_succ, label="RMD (deg)", color='tab:orange', s=20)
    plt.xlabel("Time (s)")
    plt.ylabel("RMD (deg)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("accuracy_plot_succ.png")
    print("📊 Saved plot to accuracy_plot.png")
    plt.show()  

if __name__ == "__main__":
    main()
