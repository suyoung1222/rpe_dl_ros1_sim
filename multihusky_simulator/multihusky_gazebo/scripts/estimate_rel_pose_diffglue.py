#!/usr/bin/env python
import random
from time import time

import cv2
import numpy as np
import torch

# LightGlue + SuperPoint
from lightglue import SuperPoint
from models.matching import Matching

from torchvision.transforms import ToTensor


def frame2tensor(image, device):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # Normalize to [0,1] if it is in [0..255]
    if image.max() > 1.0:
        image /= 255.0
    return torch.unsqueeze(ToTensor()(image), 0).to(device)


def process_resize(w, h, resize):
    if len(resize) == 1 and resize[0] > 0:
        scale = resize[0] / float(max(h, w))
        w_new = int(round(w * scale))
        h_new = int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        # No resizing
        w_new, h_new = w, h
    elif len(resize) == 2:
        w_new, h_new = resize[0], resize[1]
    else:
        raise ValueError("Invalid resize specification.")
    return w_new, h_new


def backproject_pixel_to_3d(x, y, depth_map, K):
    h, w = depth_map.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    z = depth_map[y, x]
    # Skip invalid or missing depth
    if z <= 0.0:
        return None
    # Homogeneous pixel coords * depth
    pt_h = np.array([x * z, y * z, z], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    pt_cam = K_inv @ pt_h  # 3D point in the follower camera
    return pt_cam


def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    image = cv2.resize(image.astype("float32"), (w_new, h_new))

    inp = frame2tensor(image, device)
    return image, inp, scales


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)

matcher_config = {
        "superpoint": {
            "nms_radius": 3,
            "keypoint_threshold": 0.0002, #0.005,
            "max_keypoints": 2048, #1024,
        },
    }
matcher = Matching(matcher_config).eval().to(device)

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        c = [1, 0, 0] if random.random() < 0.5 else [0, 1, 0]
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out



def estimate_relative_pose(
    follower_img: np.ndarray,  # Grayscale image from follower (H x W)
    leader_img: np.ndarray,  # Grayscale image from leader   (H x W)
    follower_depth: np.ndarray,  # Depth image from follower, same size as follower_img, in meters
    K_follower: np.ndarray = None,
    K_leader: np.ndarray = None,
    resize: list = [640, 480],
    max_keypoints: int = 1024,
):
    """
    Estimate the relative pose (R, t) of follower camera with respect to the leader camera

    Args:
        follower_img (np.ndarray): Grayscale image from the follower camera, shape (H, W).
        leader_img   (np.ndarray): Grayscale image from the leader camera,   shape (H, W).
        follower_depth (np.ndarray): Depth image aligned with follower_img, same shape (H, W), in meters.
        K_follower (np.ndarray): 3x3 intrinsics for the follower camera.
                                 If None, uses default placeholders.
        K_leader   (np.ndarray): 3x3 intrinsics for the leader camera.
                                 If None, uses default placeholders.
        resize (list): Desired resize. E.g. [640, 480], or [600], or [-1] for no resize.
        max_keypoints (int): Maximum number of keypoints for SuperPoint.

    Returns:
        R (np.ndarray): 3x3 rotation matrix for the transform (follower -> leader).
        t (np.ndarray): 3x1 translation vector for the transform (follower -> leader).

    Raises:
        ValueError: If not enough 3D-2D correspondences are found or PnP fails.
    """

    if K_follower is None:
        # Example intrinsics (placeholders!) for the follower camera
        fx_f = 385.6
        fy_f = 385.6
        cx_f = 326.7
        cy_f = 238.0
        K_follower = np.array(
            [[fx_f, 0.0, cx_f], [0.0, fy_f, cy_f], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    if K_leader is None:
        fx_l = 385.6
        fy_l = 385.6
        cx_l = 326.7
        cy_l = 238.0
        K_leader = np.array(
            [[fx_l, 0.0, cx_l], [0.0, fy_l, cy_l], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    

    
    

    h_f, w_f = follower_img.shape[:2]
    h_l, w_l = leader_img.shape[:2]
    w_new_f, h_new_f = process_resize(w_f, h_f, resize)
    w_new_l, h_new_l = process_resize(w_l, h_l, resize)

    # Resize the images
    follower_img_resized = cv2.resize(follower_img, (w_new_f, h_new_f)).astype(
        np.float32
    )
    leader_img_resized = cv2.resize(leader_img, (w_new_l, h_new_l)).astype(
        np.float32
    )

    # Keep track of scaling factors for backprojection or 2D correspondences
    scale_f_x = float(w_f) / float(w_new_f)
    scale_f_y = float(h_f) / float(h_new_f)
    scale_l_x = float(w_l) / float(w_new_l)
    scale_l_y = float(h_l) / float(h_new_l)

    # Convert to torch
    inp_f = frame2tensor(follower_img_resized, device)  # Follower
    inp_l = frame2tensor(leader_img_resized, device)  # Leader

    # ----------------------------------------------------------------
    # 4) Extract keypoints with SuperPoint
    # ----------------------------------------------------------------
    feats_f = extractor.extract(inp_f)  # follower feats
    feats_l = extractor.extract(inp_l)  # leader feats

    # ----------------------------------------------------------------
    # 5) Match keypoints with LightGlue (or the "Matching" from models)
    # ----------------------------------------------------------------
    pred = matcher({"image0": inp_f, "image1": inp_l})
    kpts_f = pred["keypoints0"][0].cpu().numpy()  # (N,2)
    kpts_l = pred["keypoints1"][0].cpu().numpy()  # (N,2)
    matches = pred["matches0"][0].cpu().numpy()  # (N,)
    # matching_scores = pred['matching_scores0'][0].cpu().numpy() # confidence (optional)

    valid_mask = matches > -1
    matched_kpts_f = kpts_f[valid_mask]
    matched_kpts_l = kpts_l[matches[valid_mask]]
    print(f"number of Valid matches: {len(matched_kpts_f)}")
    if len(matched_kpts_f) < 6:
        raise ValueError(
            f"Not enough matches to attempt PnP! Found {len(matched_kpts_f)}."
        )
    text = [
        'DiffGlue',
        'Keypoints: {}:{}'.format(len(kpts_l), len(kpts_f)),
        'Matches: {}'.format(len(matched_kpts_f))
    ]
    # out = make_matching_plot_fast(
    #     leader_img_resized*255, follower_img_resized*255, kpts_l, kpts_f, matched_kpts_l, matched_kpts_f, text=text,
    #     path=None, show_keypoints=True)

    # cv2.imwrite(f'log_images/{time()}.jpg', out)

    pts_3d_follower = []
    pts_2d_leader = []

    for i in range(len(matched_kpts_f)):
        # Follower pixel (resized -> original)
        x_f_res, y_f_res = matched_kpts_f[i]
        x_f = x_f_res * scale_f_x
        y_f = y_f_res * scale_f_y
        x_f_int = int(round(x_f))
        y_f_int = int(round(y_f))

        # Depth-based backprojection in the follower frame
        pt3d = backproject_pixel_to_3d(
            x_f_int, y_f_int, follower_depth, K_follower
        )
        if pt3d is None:
            continue

        # Leader pixel (resized -> original)
        x_l_res, y_l_res = matched_kpts_l[i]
        x_l = x_l_res * scale_l_x
        y_l = y_l_res * scale_l_y

        pts_3d_follower.append(pt3d)
        pts_2d_leader.append([x_l, y_l])

    if len(pts_3d_follower) < 6:
        raise ValueError(
            f"Not enough valid 3D-2D correspondences after filtering: {len(pts_3d_follower)}."
        )
    print(f"Number of feature matches after filtering: {len(pts_3d_follower)}.")

    pts_3d_follower = np.array(pts_3d_follower, dtype=np.float32)
    pts_2d_leader = np.array(pts_2d_leader, dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d_follower,
        pts_2d_leader,
        K_leader,
        distCoeffs=None,
        reprojectionError= 1.0, #3.0,
        confidence=0.985, #0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    pose_estimation_valid = 1
    if not success:
        pose_estimation_valid = 0
        raise ValueError("PnP solver failed to find a valid pose.")

    # Convert rotation vector to a 3x3 rotation matrix
    R_f2l, _ = cv2.Rodrigues(rvec)  # follower->leader
    t_f2l = tvec.reshape(3)
    
    # ----------------------------------------------------------------
    # 8) Return results
    # ----------------------------------------------------------------
    return R_f2l, t_f2l, pose_estimation_valid


if __name__ == "__main__":
    # Example usage / quick test (NOT a full demo):
    # You would replace these with actual images loaded from disk
    # or from ROS messages as np.ndarray (grayscale & depth).
    import sys

    # Dummy synthetic data for demonstration:
    # (In a real scenario, load real images + depth + intrinsics.)
    follower_gray = np.zeros((480, 640), dtype=np.uint8)
    leader_gray = np.zeros((480, 640), dtype=np.uint8)
    follower_depth = (
        np.ones((480, 640), dtype=np.float32) * 2.0
    )  # 2 meters everywhere

    try:
        R, t = estimate_relative_pose(
            follower_gray, leader_gray, follower_depth
        )
        print("Estimated rotation:\n", R)
        print("Estimated translation:\n", t)
    except ValueError as e:
        print("Estimation failed:", e)
