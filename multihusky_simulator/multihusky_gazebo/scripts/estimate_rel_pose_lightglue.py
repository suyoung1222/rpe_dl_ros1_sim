import torch
import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

torch.set_grad_enabled(False)

# Hardcoded intrinsics (adjust or load dynamically as needed)
K = np.array([[642.78, 0, 651.22],
              [0, 642.25, 356.69],
              [0, 0, 1]])

# Load LightGlue + SuperPoint models once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

def preprocess_cv_image(image_cv):
    """Converts OpenCV grayscale image to torch tensor for LightGlue."""
    img = image_cv.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).float()[None, None].to(device)
    return tensor

def estimate_relative_pose(img0_cv, img1_cv):
    """
    Estimate relative pose (R, t) between img0 and img1.
    
    Args:
        img0_cv (np.ndarray): Follower image (grayscale OpenCV format).
        img1_cv (np.ndarray): Leader image (grayscale OpenCV format).
    
    Returns:
        R (3x3 np.ndarray): Rotation matrix.
        t (3x1 np.ndarray): Translation vector.
    """
    # Convert to LightGlue input
    inp0 = preprocess_cv_image(img0_cv)
    inp1 = preprocess_cv_image(img1_cv)

    # Feature extraction
    feats0 = extractor.extract(inp0)
    feats1 = extractor.extract(inp1)

    # Matching
    matches = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]
    
    match_indices = matches['matches']  # (K, 2)
    if match_indices.shape[0] < 8:
        raise ValueError("Too few matches for pose estimation")

    kpts0 = feats0['keypoints'][match_indices[:, 0]].cpu().numpy()
    kpts1 = feats1['keypoints'][match_indices[:, 1]].cpu().numpy()

    # Normalize keypoints
    kpts0_norm = cv2.undistortPoints(kpts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    kpts1_norm = cv2.undistortPoints(kpts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    # Essential matrix
    E, mask = cv2.findEssentialMat(kpts0_norm, kpts1_norm, focal=1.0, pp=(0., 0.),
                                   method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    if E is None:
        raise ValueError("Essential matrix estimation failed")

    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, kpts0_norm, kpts1_norm)

    return R, t
