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

# Dummy placeholders (replace with your actual implementation)
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
    parser.add_argument("--est_topic", type=str, default="/h2/estimated_pose")
    parser.add_argument("--gt_topic", type=str, default="/gazebo/model_states")
    args = parser.parse_args()

    est_poses = []  # (timestamp, pose [x,y,z,qx,qy,qz,qw])
    gt_poses = []   # (timestamp, pose [x,y,z,qx,qy,qz,qw]) from Husky_h2 w.r.t. Husky_h1

    with rosbag.Bag(args.bagfile, "r") as bag:
        model_states_msgs = []
        for topic, msg, t in bag.read_messages():
            if topic == args.est_topic:
                est_pose = parse_pose(msg.pose)
                est_poses.append((msg.header.stamp, est_pose))
            elif topic == args.gt_topic:
                model_states_msgs.append((t, msg))

    if not est_poses:
        print("No data found for estimation")
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
