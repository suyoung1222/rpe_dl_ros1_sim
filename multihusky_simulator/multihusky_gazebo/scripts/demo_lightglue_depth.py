import argparse
from pathlib import Path
import torch
import numpy as np
import time
import cv2
import pdb
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

torch.set_grad_enabled(False)

def backproject_pixel_to_3d(x, y, depth_map, K):
    """Backproject a single 2D pixel (x, y) with its depth into a 3D point
       in the camera frame given the intrinsics K.
       depth_map is a 2D array with depth in *meters*, same size as the image.
    """
    # Safeguards
    if x < 0 or y < 0 or y >= depth_map.shape[0] or x >= depth_map.shape[1]:
        return None
    
    z = depth_map[y, x]
    if z <= 0.0:
        return None  # Invalid or missing depth

    # Construct homogenous pixel coords, multiply by depth
    pt_h = np.array([x * z, y * z, z], dtype=np.float32)
    
    # Invert K once outside of loops in real code for efficiency, but for clarity:
    K_inv = np.linalg.inv(K)
    
    # 3D point in the camera (follower) frame
    pt_camera = K_inv @ pt_h
    return pt_camera  # (3,)

def load_image(path, resize=None, device='cpu'):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image from path: {path}")

    img = img.astype(np.float32) / 255.0  # normalize to [0,1]

    if resize:
        if len(resize) == 1:
            scale = resize[0] / max(img.shape)
            new_w, new_h = int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))
        elif len(resize) == 2:
            new_w, new_h = resize
        else:
            raise ValueError("Resize must be one or two integers.")
        img = cv2.resize(img, (new_w, new_h))

    tensor = torch.from_numpy(img).float()[None, None]  # shape: [1, 1, H, W]
    return tensor.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image0', type=str, default='../bagfiles/loop1_robot2/1744928745.213028.png',
                        help='Leader image')
    parser.add_argument('--image1', type=str, default='../bagfiles/loop1_robot2/1744928847.723895.png',
                        help='Follower image')
    parser.add_argument('--depth1', type=str, default='../bagfiles/loop1_robot2_depth/1744928847798241238.png',
                        help='Depth image corresponding to the follower image1')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--max_keypoints', type=int, default=1024)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize LightGlue with SuperPoint features
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # Load images
    img0 = load_image(args.image0, resize=args.resize, device=device)  # Leader
    img1 = load_image(args.image1, resize=args.resize, device=device)  # Follower

    # 1) Feature extraction
    start_extract = time.time()
    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)
    end_extract = time.time()
    print(f"Feature extraction time: {end_extract - start_extract:.4f}s")

    # 2) Feature matching
    start_match = time.time()
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    # rbd -> 'remove batch dimension'
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # shape (N, 2)

    points0 = feats0['keypoints'][matches[:, 0]]  # (N, 2)
    points1 = feats1['keypoints'][matches[:, 1]]  # (N, 2)
    conf = matches01.get('confidence', torch.ones(len(matches)))

    mkpts0 = points0.cpu().numpy()  # Leader image keypoints
    mkpts1 = points1.cpu().numpy()  # Follower image keypoints
    mconf = conf.cpu().numpy()

    end_match = time.time()
    print(f"Valid matches: {len(mkpts0)}")
    print(f"Matching time: {end_match - start_match:.4f}s")

    # -------------------------------------------------------------
    # 3) Direct pose estimation with depth (scaled) using solvePnPRansac
    # -------------------------------------------------------------
    start_pose = time.time()

    # Intrinsics (leader camera). 
    # If the leader & follower have different intrinsics, define each accordingly.
    fx_leader = 642.78
    fy_leader = 642.25
    cx_leader = 651.22
    cy_leader = 356.69
    K_leader = np.array([[fx_leader, 0,         cx_leader],
                         [0,         fy_leader, cy_leader],
                         [0,         0,         1]], dtype=np.float32)

    # Intrinsics for the follower camera (used to back-project 2D + depth -> 3D).
    # If it's the same camera model, you can reuse the same K. Otherwise, define a new one.
    fx_follower = 642.78
    fy_follower = 642.25
    cx_follower = 651.22
    cy_follower = 356.69
    K_follower = np.array([[fx_follower, 0,           cx_follower],
                           [0,           fy_follower, cy_follower],
                           [0,           0,           1]], dtype=np.float32)

    # Load the depth image for the follower
    depth_img1 = cv2.imread(args.depth1, cv2.IMREAD_UNCHANGED)
    if depth_img1 is None:
        raise ValueError(f"Cannot read depth image from path: {args.depth1}")
    depth_img1 = depth_img1.astype(np.float32) / 1000000.0  # convert mm->m
    # Build 3D->2D correspondences: 3D from follower's frame, 2D in leader's image
    pts_3d_follower = []
    pts_2d_leader = []
    pdb.set_trace()
    for i in range(len(mkpts1)):
        x_f = mkpts1[i, 0]
        y_f = mkpts1[i, 1]
        x_f_rounded = int(round(x_f))
        y_f_rounded = int(round(y_f))

        # Backproject in follower frame (returns a (3,) point or None)
        pt3d = backproject_pixel_to_3d(x_f_rounded, y_f_rounded, depth_img1, K_follower)
        if pt3d is None:
            continue

        # 2D point in leader image
        x_l = mkpts0[i, 0]
        y_l = mkpts0[i, 1]

        pts_3d_follower.append(pt3d)
        pts_2d_leader.append([x_l, y_l])

    if len(pts_3d_follower) < 6:
        print("Not enough valid 3D-2D correspondences for PnP!")
    else:
        pts_3d_follower = np.array(pts_3d_follower, dtype=np.float32)
        pts_2d_leader   = np.array(pts_2d_leader,   dtype=np.float32)

        # Solve PnP to get transformation from the follower's 3D points into the leader camera frame
        # You can use solvePnP or solvePnPRansac. RANSAC is more robust but slower.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_follower,
            pts_2d_leader,
            K_leader,
            distCoeffs=None,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("solvePnPRansac failed to find a valid pose.")
        else:
            # Convert rvec to 3x3 rotation matrix
            R_f2l, _ = cv2.Rodrigues(rvec)
            t_f2l = tvec.reshape(-1)

            # 4x4 transformation from follower frame to leader frame
            M_f2l = np.eye(4, dtype=np.float32)
            M_f2l[:3, :3] = R_f2l
            M_f2l[:3,  3] = t_f2l

            print("========== SCALED RELATIVE POSE (Follower -> Leader) ==========")
            print("Rotation matrix R_f2l:\n", R_f2l)
            print("Translation t_f2l (meters):", t_f2l)
            print("4x4 transform M_f2l:\n", M_f2l)
            print(f"Inliers (PnP): {len(inliers)} / {len(pts_3d_follower)}")

    end_pose = time.time()
    print(f"Pose estimation time (PnP w/ depth): {end_pose - start_pose:.4f}s")

    # Optionally save the raw matches somewhere
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'matches_lightglue.npz',
                        mkpts0=mkpts0, mkpts1=mkpts1,
                        confidence=mconf,
                        all_kpts0=feats0['keypoints'].cpu().numpy(),
                        all_kpts1=feats1['keypoints'].cpu().numpy())
    print(f"Saved LightGlue matches to {output_dir / 'matches_lightglue.npz'}")

if __name__ == '__main__':
    main()
