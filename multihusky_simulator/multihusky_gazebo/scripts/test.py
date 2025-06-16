import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_SE3(position, quaternion):
    rot = R.from_quat(quaternion)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T

# parser.add_argument('--image0', type=str, default='../bagfiles/tum/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png',
#                         help='Leader image')
# 1305031102.1758 1.3405 0.6266 1.6575 0.6574 0.6126 -0.2949 -0.3248
# timestamp tx ty tz qx qy qz qw

#     parser.add_argument('--image1', type=str, default='../bagfiles/tum/rgbd_dataset_freiburg1_xyz/rgb/1305031114.179337.png',
#                         help='Follower image')
# 1305031114.1757 1.2599 0.4174 1.5860 0.6274 0.6536 -0.2868 -0.3113

#     parser.add_argument('--depth1', type=str, default='../bagfiles/tum/rgbd_dataset_freiburg1_xyz/depth/1305031114.174114.png',

# Robot 1
p1 = [1.3405, 0.6266, 1.6575]
q1 = [0.6574, 0.6126, -0.2949, -0.32485]
T1 = pose_to_SE3(p1, q1)

# Robot 2
p2 = [1.2599, 0.4174, 1.5860]
q2 = [0.6274, 0.6536, -0.2868, -0.3113]
T2 = pose_to_SE3(p2, q2)

# Relative transforms
T1_wrt_2 = np.linalg.inv(T2) @ T1
T2_wrt_1 = np.linalg.inv(T1) @ T2

np.set_printoptions(precision=6, suppress=True)
print("Robot1 w.r.t Robot2 (T1_wrt_2):\n", T1_wrt_2)
print("Robot2 w.r.t Robot1 (T2_wrt_1):\n", T2_wrt_1)

# TUM
# Robot1 w.r.t Robot2 (T1_wrt_2):
#  [[ 0.994869 -0.086997  0.05164   0.210771]
#  [ 0.088628  0.995608 -0.030175  0.007063]
#  [-0.048788  0.034597  0.99821  -0.104398]
#  [ 0.        0.        0.        1.      ]]
# Robot2 w.r.t Robot1 (T2_wrt_1): 1->2
#  [[ 0.994869  0.088628 -0.048788 -0.215409]
#  [-0.086997  0.995608  0.034597  0.014916]
#  [ 0.05164  -0.030175  0.99821   0.09354 ]
#  [ 0.        0.        0.        1.      ]]