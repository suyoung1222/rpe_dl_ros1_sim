import argparse
import numpy as np
import cv2
import torch
import pdb

from models.matching import Matching
torch.set_grad_enabled(False)

import random

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


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
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
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
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


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DiffGlue demo with relative pose estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1600],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--max_keypoints', type=int, default=2048,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
    }
    
    set_seed(0)
    
    matching = Matching(config).eval().to(device)

    # Load the image pair.
    # image0, inp0, scales0 = read_image('../bagfiles/loop1_robot1/1744928847.735512.png', device, opt.resize)
    # image1, inp1, scales1 = read_image('../bagfiles/loop1_robot2/1744928847.723895.png', device, opt.resize)
    image0, inp0, scales0 = read_image('/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/demo/images/test_img1_occ.jpg', device, opt.resize)
    image1, inp1, scales1 = read_image('/work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/DiffGlue/demo/images/test_img2.jpg', device, opt.resize)

    pred = matching({'image0': inp0, 'image1': inp1})
    
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = error_colormap(confidence[valid])
    text = [
        'DiffGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    out = make_matching_plot_fast(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True)

    cv2.imwrite('test.jpg', out)
 
    print("Matching Finished.")

    ##### Relative pose estimation based on matches using opencv
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
            inlier_color = color[inliers]

            # Generate new text to annotate the image
            text_inlier = [
                'DiffGlue + RANSAC',
                f'Inlier Matches: {len(inlier_mkpts0)} / {len(mkpts0)}'
            ]

            # Visualize inlier matches only
            out_ransac = make_matching_plot_fast(
                image0, image1, kpts0, kpts1, inlier_mkpts0, inlier_mkpts1, inlier_color,
                text_inlier, path='test_ransac.jpg',
                show_keypoints=True)

            print("Saved inlier visualization with RANSAC to ./images/test_img1_img2_ransac.jpg")


        print("Estimated Rotation:\n", R)
        print("Estimated Translation:\n", t)