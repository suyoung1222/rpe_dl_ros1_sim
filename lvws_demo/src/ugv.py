#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from math import pi, tau, dist, fabs, cosh
import math

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

from collections import deque

import argparse

# 10 Hz setting
eps_angular = .05
eps_d = .3
max_linear_velocity = 2
max_angular_velocity = 2
Kpa = 3
Kpv = 2
rospy_rate = 10

namespace = None


def poseCallback(message):
    global last_pose
    idx = message.name.index('{}{}'.format('Husky_' if namespace[:5]=='husky' else '', namespace))
    last_pose = message.pose[idx]

def cmdCallback(message):
    global last_cmd_pose, cmd_queue
    last_cmd_pose = message
    if len(cmd_queue) > 0:
        latest_pose = cmd_queue[0]
        if (math.sqrt((latest_pose.x-last_cmd_pose.x)**2+(latest_pose.y-last_cmd_pose.y)**2+(latest_pose.theta-last_cmd_pose.theta)**2)<0.01):
            return
    cmd_queue.appendleft(last_cmd_pose)

def run(waypoint):
    global last_pose
    if rospy.is_shutdown():
        return

    rospy.loginfo('{}: ({:.2f},{:.2f},{:.2f})'.format(namespace, waypoint[0], waypoint[1], waypoint[2]))
    curr_pose = last_pose
    target_theta = math.atan2(waypoint[1] - curr_pose.position.y, waypoint[0] - curr_pose.position.x)
    
    turn(target_theta)
    rospy.sleep(0.1)
    
    move(waypoint)
    rospy.sleep(0.1)
    
    turn(waypoint[2])
    rospy.sleep(0.1)


def turn(target_theta):
    global rate, publisher, last_pose, eps_angular, Kpa
    # compute target angle based on current coord and next waypoint coord

    # loop until correct angle has been achieved
    while not rospy.is_shutdown():
        # compute angular distance theta_err
        curr_pose = last_pose
        theta = euler_from_quaternion([curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w])[2]
        theta_err = theta - target_theta
        theta_err = (theta_err + math.pi) % math.tau - math.pi    # get shortest angle

        # rospy.loginfo('{}, {}, {}'.format(theta_err, theta, target_theta))
        if abs(theta_err) < eps_angular:
            break

        # compute angular_velocity based on Kpa * theta_err
        angular_velocity = -Kpa * theta_err
        angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity))

        # send angular_velocity command
        cmd = Twist()
        cmd.angular.z = angular_velocity
        publisher.publish(cmd)

        rate.sleep()

    # send stop command
    cmd = Twist()
    cmd.angular.z = 0
    publisher.publish(cmd)
    return


def move(waypoint):
    global rate, publisher, last_pose, eps_angular, eps_d, Kpa, Kpv
    # loop until correct position and angle has been achieved
    while not rospy.is_shutdown():
        # compute target angle based on current coord and next waypoint coord
        curr_pose = last_pose
        target_theta = math.atan2(waypoint[1] - curr_pose.position.y, waypoint[0] - curr_pose.position.x)

        # compute angular distance theta_err and distance error d_err
        curr_pose = last_pose
        # rospy.loginfo("curr_pose: {}".format(curr_pose.position))
        # rospy.loginfo("waypoint: {}".format(waypoint))
        theta = euler_from_quaternion([curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w])[2]
        theta_err = theta - target_theta
        theta_err = (theta_err + math.pi) % math.tau - math.pi    # get shortest angle
        d_err = math.sqrt((curr_pose.position.y - waypoint[1])**2 + (curr_pose.position.x - waypoint[0])**2)

        # if angle is greater than pi/2, set angle in opposite direction and move backwards instead
        # if abs(theta_err) > math.pi/2:
        #     theta_err % math.tau - math.pi
        #     d_err = -d_err

        if abs(d_err) < eps_d: 
            break

        # compute linear_velocity output based on Kpv * d_err
        linear_velocity = Kpv * d_err
        linear_velocity = max(-max_linear_velocity, min(max_linear_velocity, linear_velocity))
        # rospy.loginfo('{}, {}'.format(d_err, linear_velocity))
        # compute angular_velocity based on Kpa * theta_err
        angular_velocity = -Kpa * theta_err
        angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity))

        # send linear_velocity and angular_velocity command
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        publisher.publish(cmd)
        
        rate.sleep()

    # send stop command
    cmd = Twist()
    cmd.linear.x = 0
    cmd.angular.z = 0
    publisher.publish(cmd)
    return



if __name__ == '__main__':
    global rate, publisher
    global last_pose, last_cmd_pose
    global cmd_queue
    last_pose = None
    last_cmd_pose = None

    if len(sys.argv) < 2:
        exit()
    
    namespace = sys.argv[1]
    
    if namespace[:5]=='husky':
        max_linear_velocity *= 1.5

    # parser = argparse.ArgumentParser()
    # parser.add_argument('namespace', type=str)
    # parser.add_argument('namespace', type=str)

    # args = parser.parse_args()

    # namespace = args.namespace


    namespace_to_id = dict()
    namespace_to_id['jackal1'] = 0
    namespace_to_id['jackal2'] = 1
    namespace_to_id['husky1'] = 2
    
    try:
        rospy.init_node('{}_move'.format(namespace), anonymous=True)

        # position_topic = '/jackal1/odom_ground_truth'
        # subscriber = rospy.Subscriber(position_topic, Odometry, poseCallback)
        position_topic = '/gazebo/model_states'
        subscriber = rospy.Subscriber(position_topic, ModelStates, poseCallback)
        cmd_topic = '/scheduler/pos/{}'.format(namespace)
        subscriber_cmd = rospy.Subscriber(cmd_topic, Pose2D, cmdCallback)
        cmd_vel_topic = '/{}/cmd_vel'.format(namespace)
        publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        status_topic = '/scheduler/status'
        publisher_status = rospy.Publisher(status_topic, String, queue_size=10)
        rate = rospy.Rate(rospy_rate)
        
        cmd_queue = deque()
        
        while last_pose is None or last_cmd_pose is None: # wait until initial pose has been set
            rate.sleep()

        print('start')

        
        while not rospy.is_shutdown():
            if (len(cmd_queue) > 0):
                status_msg = String('{} start'.format(namespace))
                publisher_status.publish(status_msg)
                while (len(cmd_queue) > 0):
                    pose2d_msg = cmd_queue.pop()
                    waypoint = (pose2d_msg.x, pose2d_msg.y, pose2d_msg.theta)
                    run(waypoint)
                status_msg = String('{} end'.format(namespace))
                publisher_status.publish(status_msg)

            rate.sleep()
        
    except rospy.ROSInterruptException as e: 
        rospy.logerr(e)
        exit(1)


        cmd = Twist()
        cmd.angular.z = angular_velocity
        publisher.publish(cmd)


        curr_odom = last_odom
        curr_pose_yaw = euler_from_quaternion([curr_odom.orientation.x, curr_odom.orientation.y, curr_odom.orientation.z, curr_odom.orientation.w])[2]
        prev_pose = curr_pose = (curr_odom.position.x, curr_odom.position.y, curr_pose_yaw)
        motion_pose = prev_pose