#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from images import RosImageCollector
from tf.transformations import euler_from_quaternion
from torchvision import transforms

bridge = CvBridge()

image_log_dir = os.path.join(os.path.dirname(__file__), "image_logs")
os.makedirs(image_log_dir, exist_ok=True)

model_dir = os.path.dirname(__file__)
model_dir = os.path.join(model_dir, "..", "models")

enc_model_path = os.path.join(model_dir, "0kc5po4ee18_float32_jit_cpu_enc.ts")
msg_model_path = os.path.join(model_dir, "0kc5po4ee18_float32_jit_cpu_msg.ts")
post_model_path = os.path.join(
    model_dir, "0kc5po4ee18_float32_jit_cpu_post.ts"
)


# TODO: try half precision


class CovisNetInferencer:
    COUNTER = 0

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        rospy.init_node("covisnet_inferencer")
        self.rate = rospy.Rate(10)

        self.collector = RosImageCollector()
        self.rate = rospy.Rate(10)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    224,
                    antialias=True,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        self.relative_pose_pub = rospy.Publisher(
            "/relative_poses", PoseArray, queue_size=1
        )

        self.enc = torch.jit.load(enc_model_path)
        # self.enc = torch.jit.optimize_for_inference(self.enc)
        # self.enc = torch.jit.freeze(self.enc)
        # self.enc = torch.jit.trace(self.enc, torch.randn(1, 3, 224, 224).to(self.device))

        self.msg = torch.jit.load(msg_model_path)
        # self.msg = torch.jit.optimize_for_inference(self.msg)
        # self.msg = torch.jit.freeze(self.msg)
        # self.msg = torch.jit.trace(self.msg, (torch.randn(1, 3, 224, 224).to(self.device), torch.randn(1, 512).to(self.device)))

        self.post = torch.jit.load(post_model_path)
        # self.post = torch.jit.optimize_for_inference(self.post)
        # self.post = torch.jit.freeze(self.post)
        # self.post = torch.jit.trace(self.post, (torch.randn(1, 512).to(self.device), torch.randn(1, 512).to(self.device)))

        self.enc.eval()
        self.msg.eval()
        self.post.eval()

        self.enc.to(self.device)
        self.msg.to(self.device)
        self.post.to(self.device)

        rospy.loginfo("CoViS-Net Inferencer initialized")

        # rospy.spin()

    def start(self):
        while not rospy.is_shutdown():
            self.COUNTER += 1

            images = self.collector.fetch()

            if self.COUNTER % 10 == 0:
                print([(img.height, img.width) for img in images])

                cv2.imwrite(
                    os.path.join(
                        os.path.dirname(__file__),
                        image_log_dir,
                        f"img_{self.COUNTER}.png",
                    ),
                    cv2.hconcat(
                        [
                            cv2.resize(
                                bridge.imgmsg_to_cv2(
                                    img, desired_encoding="bgr8"
                                ),
                                (640, 480),
                            )
                            for img in images
                        ]
                    ),
                )
                print(f"Saved image {self.COUNTER}")

            images = [
                self.transform(
                    bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
                )
                for img in images
            ]

            self.infer(torch.stack(images, dim=0))

            # Sleep to maintain the loop rate
            self.rate.sleep()

    @torch.no_grad()
    def infer(self, batch_tensor):
        batch_tensor = batch_tensor.to(self.device)

        # start = time.time()
        # print(data["img"].shape)
        encs = self.enc(batch_tensor)
        # print(f"Encoding time: {time.time() - start:.2f}s")

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        pose_array.poses = []

        primary_frame = encs[0]

        for i, enc_i in enumerate(encs[1:], 1):
            m = self.msg(primary_frame.unsqueeze(0), enc_i.unsqueeze(0))
            pos, pos_var, heading, heading_var = self.post(m)

            pos = pos[0]
            heading = heading[0]

            pose_array.poses.append(
                Pose(
                    position=Point(pos[0][0], pos[0][1], pos[0][2]),
                    orientation=Quaternion(
                        x=heading[0],
                        y=heading[1],
                        z=heading[2],
                        w=heading[3],
                    ),
                )
            )

            print(
                f"Node {i}: pos {pos[:2].tolist()}, heading {np.rad2deg(euler_from_quaternion(heading)[2]):.2f}"
            )
            # print(f"Node {i}: pos_var {pos_var}, heading_var {heading_var}")

        self.relative_pose_pub.publish(pose_array)


if __name__ == "__main__":
    runner = CovisNetInferencer()
    try:
        runner.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
