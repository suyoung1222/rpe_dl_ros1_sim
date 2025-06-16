import argparse
from pathlib import Path
import torch
import numpy as np
import time

import cv2
from models.matching import Matching

torch.set_grad_enabled(False)
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
    parser.add_argument('--image0', type=str, default='./jackal_test/robot1.png', help='Path to first image')
    parser.add_argument('--image1', type=str, default='./jackal_test/robot2.png', help='Path to second image')
    parser.add_argument('--output_dir', type=str, default='./jackal_test/result', help='Directory to save results')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--max_keypoints', type=int, default=-1)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
            'weights': args.superglue,
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    # Load and resize images
    img0 = load_image(args.image0, resize=args.resize).to(device)
    img1 = load_image(args.image1, resize=args.resize).to(device)

    # Run matching
    start = time.time()
    data = matching({'image0': img0, 'image1': img1})
    end = time.time()

    print(f"Matching time: {end - start:.4f}s")

    kpts0 = data['keypoints0'][0].cpu().numpy()
    kpts1 = data['keypoints1'][0].cpu().numpy()
    matches = data['matches0'][0].cpu().numpy()
    confidence = data['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = confidence[valid]

    print(f"Valid matches: {len(mkpts0)}")


    ##### Relative pose estimation based on matches using opencv
    start_pose = time.time()
    fx = 642.78
    fy = 642.25
    cx = 651.22
    cy = 356.69
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    D = np.array([-0.0564, 0.06635, 0.0004104, 0.00080395, -0.02115]) # k1​,k2​,t1​,t2​,k3

    # Normalize matched keypoints
    mkpts0_norm = cv2.undistortPoints(mkpts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    mkpts1_norm = cv2.undistortPoints(mkpts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    # Estimate Essential matrix
    E, mask = cv2.findEssentialMat(mkpts0_norm, mkpts1_norm, focal=1.0, pp=(0., 0.),
                                   method=cv2.RANSAC, prob=0.999, threshold=1e-3)

    if E is None:
        print("Essential matrix estimation failed.")
    else:
        # Recover relative pose
        points, R, t, mask_pose = cv2.recoverPose(E, mkpts0_norm, mkpts1_norm)
        print(f"Recovered {points} inliers.")

        if E is None or R is None or t is None:
            print("Pose recovery failed.")
        else:
            print("Estimated Relative Pose (camera0 to camera1):")
            print("Rotation matrix R:")
            print(R)

            print("Translation vector t (up to scale):")
            print(t.ravel())  # Flatten to a row for printing

            # Optionally compute a 4x4 transformation matrix
            M_0to1 = np.eye(4)
            M_0to1[:3, :3] = R
            M_0to1[:3, 3] = t.ravel()
            print("Estimated 4x4 transformation matrix M_0to1:")
            print(M_0to1)

            # Also count inliers
            inliers = mask_pose.ravel().astype(bool)
            print(f"Number of inlier matches: {np.sum(inliers)} / {len(mkpts0)}")

            # Filter inlier matched keypoints based on RANSAC
            inlier_mkpts0 = mkpts0[inliers]
            inlier_mkpts1 = mkpts1[inliers]

    
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:\n", t)
    end_pose = time.time()

    print(f"Pose estimation time: {end_pose - start_pose:.4f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'matches.npz',
                        mkpts0=mkpts0, mkpts1=mkpts1,
                        matches=matches, confidence=confidence,
                        all_kpts0=kpts0, all_kpts1=kpts1)

    print(f"Saved matches to {output_dir / 'matches.npz'}")

if __name__ == '__main__':
    main()