import argparse
from pathlib import Path
import torch
import numpy as np
import time
import cv2
# from models.diffglue import DiffGlueMatcher  # from DiffGlue repo
from models.matching import Matching
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import random

torch.set_grad_enabled(False)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    inp = frame2tensor(image, device)
    return image, inp, scales

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
    parser.add_argument('--image0', type=str, default='../bagfiles/loop1_robot1/1744928847.735512.png')
    parser.add_argument('--image1', type=str, default='../bagfiles/loop1_robot2/1744928847.723895.png')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--max_keypoints', type=int, default=1024)
    args = parser.parse_args()

    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': 0.005,
            'max_keypoints': 512
        },
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = Matching(config).eval().to(device)

    # img0 = load_image(args.image0, resize=args.resize, device=device)
    # img1 = load_image(args.image1, resize=args.resize, device=device)
    img0, inp0, scales0 = read_image(args.image0, device,  args.resize) # Follower
    img1, inp1, scales1 = read_image(args.image1, device,  args.resize) # Leader
 
    # Extract features
    start = time.time()
    feats0 = extractor.extract(inp0)
    feats1 = extractor.extract(inp1)
    end = time.time()

    print(f"Extraction time: {end - start:.4f}s")

    # Make sure descriptors are present
    if 'descriptors' not in feats0 or 'descriptors' not in feats1:
        raise ValueError("Descriptors missing. Ensure SuperPoint outputs descriptors.")

    # # Run DiffGlue matching
    # start = time.time()
    # data = matcher({
    #     'keypoints0': feats0['keypoints'],
    #     'keypoints1': feats1['keypoints'],
    #     'descriptors0': feats0['descriptors'],
    #     'descriptors1': feats1['descriptors'],
    # })
    # end = time.time()

    # print(f"Matching time: {end - start:.4f}s")
    start_match = time.time()
    pred = matcher({'image0': inp0, 'image1': inp1})
    end_match = time.time()
    
    
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    print("Matching Finished.")
    print(f"Matching time: {end_match - start_match:.4f}s")
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



    # save matches
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'matches_lightglue.npz',
                        mkpts0=mkpts0, mkpts1=mkpts1,
                        confidence=confidence,
                        all_kpts0=feats0['keypoints'].cpu().numpy(),
                        all_kpts1=feats1['keypoints'].cpu().numpy())

    print(f"Saved LightGlue matches to {output_dir / 'matches_lightglue.npz'}")

if __name__ == '__main__':
    main()
