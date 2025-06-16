#!/usr/bin/env python3

import threading

import message_filters
import rospy
from sensor_msgs.msg import Image


class RosImageCollectorSynchronized:
    def __init__(self):
        self.mutex = threading.Lock()

        self.topics = [
            "/camera/color/image_raw",
            "/Jackal_1_colorImg",
            "/Jackal_2_colorImg",
        ]
        self.images = None

        filtered_subs = [
            message_filters.Subscriber(topic, Image) for topic in self.topics
        ]

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            filtered_subs, queue_size=5, slop=5.0, allow_headerless=True
        )
        self.synchronizer.registerCallback(self.callback)

        rospy.loginfo("RosImageCollector initialized")

    def callback(self, *args):
        print("Callback triggered")
        with self.mutex:
            self.images = args

    def wait_for_images(self):
        while self.images is None:
            rospy.sleep(0.01)

    def fetch(self, primary_first=True):
        self.wait_for_images()
        with self.mutex:
            if primary_first:
                return self.images
            else:
                return self.images[::-1]


class RosImageCollector:
    def __init__(self, standalone=False):
        self.mutex = threading.Lock()

        # ordered; first is primary
        self.topics = [
            # "/camera/color/image_raw",
            "/Jackal_1_colorImg",
            "/Jackal_2_colorImg",
        ]
        self.images = None
        self._reset()

        for i, topic in enumerate(self.topics):
            self.create_callbacks_dynamically(i)
            rospy.Subscriber(
                topic, Image, getattr(self, f"callback_{i}"), queue_size=1
            )
        rospy.loginfo("RosImageCollector initialized")

        if standalone:
            rospy.spin()

    @classmethod
    def create_callbacks_dynamically(cls, idx):
        def callback(self, msg):
            # print(f"Callback {idx} triggered")
            with self.mutex:
                self.images[idx] = msg

        callback.__name__ = f"callback_{idx}"
        setattr(cls, callback.__name__, callback)

    def _reset(self):
        self.images = [None] * len(self.topics)

    def wait_for_images(self):
        while None in self.images:
            # print(f"Waiting for images... ")
            rospy.sleep(0.05)

    def fetch(self, primary_first=True):
        self.wait_for_images()
        with self.mutex:
            if primary_first:
                images = self.images
            else:
                images = self.images[::-1]

            self._reset()
            return images


if __name__ == "__main__":

    try:
        rospy.init_node("image_collector")
        collector = RosImageCollector(True)
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            print(len(collector.fetch()))
            rate.sleep()

        # rospy.spin()

    except rospy.ROSInterruptException:
        pass
