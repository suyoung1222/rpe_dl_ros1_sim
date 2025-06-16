#!/usr/bin/env python3

# %%
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
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import numpy as np

from gazebo_msgs.srv import GetModelState, SetModelState

from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse


# %%
def statusCallback(message):
    global last_status, status_jackal1, status_jackal2, status_husky1
    print(str(message.data))
    last_status = str(message.data).split()
    assert(len(last_status) == 2)
    robot_name = last_status[0]
    robot_state = last_status[1]
    if (robot_name == 'jackal1'):
        status_jackal1 = 1 if (robot_state == 'start') else 0
    elif (robot_name == 'jackal2'):
        status_jackal2 = 1 if (robot_state == 'start') else 0
    elif (robot_name == 'husky1'):
        status_husky1 = 1 if (robot_state == 'start') else 0

# wait for 10s before starting
rospy.sleep(10)

status_jackal1 = status_jackal2 = status_husky1 = 0

moveit_commander.roscpp_initialize(sys.argv)

rospy.init_node('scheduler', anonymous=True)
rospy_rate = 10
rate = rospy.Rate(rospy_rate)

status_topic = '/scheduler/status'
subscriber_status = rospy.Subscriber(status_topic, String, statusCallback)

rospy.wait_for_service('gazebo/set_model_state')
rospy.wait_for_service('gazebo/get_model_state')
get_model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_model_coordinates = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

rospy.wait_for_service('link_attacher_node/attach')
rospy.wait_for_service('link_attacher_node/detach')
link_attach = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
link_detach = rospy.ServiceProxy('/link_attacher_node/detach', Attach)

# %%
def attach_links(model1, link1, model2, link2):
    global link_attach
    req = AttachRequest()
    req.model_name_1 = model1
    req.link_name_1 = link1
    req.model_name_2 = model2
    req.link_name_2 = link2
    link_attach.call(req)

def detach_links(model1, link1, model2, link2):
    global link_detach
    req = AttachRequest()
    req.model_name_1 = model1
    req.link_name_1 = link1
    req.model_name_2 = model2
    req.link_name_2 = link2
    link_detach.call(req)

# %%
# req = AttachRequest()
# req.model_name_1 = 'Husky_husky1'
# req.link_name_1 = 'base_link'
# req.model_name_2 = 'cardboard_box_L_1'
# req.link_name_2 = 'link'
# link_attach.call(req)

# %%

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_left = moveit_commander.MoveGroupCommander("left_arm")
group_middle = moveit_commander.MoveGroupCommander("middle_arm")
group_right = moveit_commander.MoveGroupCommander("right_arm")

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

default = [0, 0, 0, -90, 0, 0, 0]
up = [0, 0, 0, -85, 0, 85, 45]
load_robot_husky = [-75, 50, 0, -60, 0, 115, 45]
load_robot_jackal = [-75, 65, 0, -55, 0, 125, 45]
load_robot_up = [-75, 30, 0, -60, 0, 115, 45]
unload_robot_husky = [-5, 50, 0, -60, 0, 115, 45]
unload_robot_jackal = [-5, 65, 0, -55, 0, 125, 45]
unload_robot_up = [-5, 40, 0, -55, 0, 125, 45]
process1_down = [155, 60, 0, -30, 0, 130, 45]
process2_down = [75, 0, 40, -115, 0, 115, 45]
process1_up = [155, 0, 0, -70, 0, 90, 45]
process2_up = [75, 0, 40, -60, 0, 110, 45]

def move_arm(group, target, wait=False):
    joint_goal = group.get_current_joint_values()
    for i in range(len(target)):
        joint_goal[i] = target[i] * pi/180.0
    
    group.go(joint_goal, wait=wait)

def arm_process(group, process=1, wait=True):
    if process == 1:
        move_arm(group, process1_down, wait=wait)
    elif process == 2:
        move_arm(group, process2_down, wait=wait)

def arm_load1(group, process=1, wait=True):
    if process == 1:
        move_arm(group, process1_down, wait=wait)
    elif process == 2:
        move_arm(group, process2_down, wait=wait)
def arm_load2(group, process=1, wait=True):
    if process == 1:
        move_arm(group, process1_up, wait=wait)
    elif process == 2:
        move_arm(group, process2_up, wait=wait)
def arm_load3(group, process=1, wait=True):
    move_arm(group, load_robot_up, wait=wait)

def arm_load_group(group_list):
    for group,process in group_list:
        arm_load1(group, process=process)
    if process==1:
        for group,process in group_list:
            arm_load2(group, process=process)
    for group,process in group_list:
        arm_load3(group, process=process)

def arm_unload1(group, process=1, wait=True):
    # move_arm(group, unload_robot_jackal, wait=wait)
    move_arm(group, unload_robot_up, wait=wait)
def arm_unload2(group, process=1, wait=True):
    if process == 1:
        move_arm(group, process1_down, wait=wait)
    elif process == 2:
        move_arm(group, process2_down, wait=wait)

def arm_unload_group(group_list):
    for group,process in group_list:
        arm_unload1(group, process=process)
    for group,process in group_list:
        arm_unload2(group, process=process)

# %%
pos_jackal1_topic = '/scheduler/pos/jackal1'
pos_jackal2_topic = '/scheduler/pos/jackal2'
pos_husky1_topic = '/scheduler/pos/husky1'
publisher_jackal1 = rospy.Publisher(pos_jackal1_topic, Pose2D, queue_size=10)
publisher_jackal2 = rospy.Publisher(pos_jackal2_topic, Pose2D, queue_size=10)
publisher_husky1 = rospy.Publisher(pos_husky1_topic, Pose2D, queue_size=10)

# %%
ddown  = 0.0
dright = 1.5708
dup    = 3.1416
dleft  = 4.7124

L3l = -3.5 #-4.3
L2l = -3.5
L1l = -2.7
Ls = -0.5
Us = 0.9  #0.6
U3l = 1.5
U2l = 2.4
U1l = 3.3
rstart = 4.5

U1s = -6.1
L1s = -5.2
D1s = -4.4
U2s = -2.9
L2s = -2.0
D2s = -1.2
U3s = 0.3
L3s = 1.2
D3s = 2.0
L3s_offset = 0.5

AC1_load = [Ls, L1s, dup]
AC1_unload = [Us, D1s, ddown]
L1_exit = [L1l, L1s, dup]
U1_exit = [U1l, D1s, ddown]
L1_enter_U1 = [U1l, L1s, dup]
L1_enter_U2 = [U2l, L1s, dup]
L1_enter_U3 = [U3l, L1s, dup]
U1_enter1_L1 = [L1l, U1s, ddown]
U1_enter1_L2 = [L2l, U1s, ddown]
U1_enter1_L3 = [L3l, U1s, ddown]
U1_enter2 = [Us, U1s, ddown]

AC2_load = [Ls, L2s, dup]
AC2_unload = [Us, D2s, ddown]
L2_exit = [L2l, L2s, dup]
U2_exit = [U2l, D2s, ddown]
L2_enter_U1 = [U1l, L2s, dup]
L2_enter_U2 = [U2l, L2s, dup]
L2_enter_U3 = [U3l, L2s, dup]
U2_enter1_L1 = [L1l, U2s, ddown]
U2_enter1_L2 = [L2l, U2s, ddown]
U2_enter1_L3 = [L3l, U2s, ddown]
U2_enter2 = [Us, U2s, ddown]

AC3_load = [Ls, L3s, dup]
AC3_unload = [Us, D3s, ddown]
L3_exit = [L3l, L3s, dup]
U3_exit = [U3l, D3s, ddown]
L3_enter_U1 = [U1l, L3s, dup]
L3_enter_U2 = [U2l, L3s, dup]
L3_enter_U3 = [U3l, L3s, dup]
U3_enter1_L1 = [L1l, U3s, ddown]
U3_enter1_L2 = [L2l, U3s, ddown]
U3_enter1_L3 = [L3l, U3s, ddown]
U3_enter2 = [Us, U3s, ddown]
AC3_load_team = [Ls, L3s_offset, dup]

start_jackal1 = [rstart, -1.9, dup]
start_jackal2 = [rstart, 0.2, dup]
start_husky1 = [rstart, 2.3, dup]

start_front_jackal1 = [U1l, -1.9, dup]
start_front_jackal2 = [U1l, 0.2, dup]
start_front_husky1 = [U1l, 2.3, dup]

pos = ['start', 'start', 'start']

publisher_to_pos_idx = dict()
publisher_to_pos_idx[publisher_jackal1] = 0
publisher_to_pos_idx[publisher_jackal2] = 1
publisher_to_pos_idx[publisher_husky1] = 2

def move_ugv(publisher, target):
    cmd = Pose2D()
    cmd.x = target[0]
    cmd.y = target[1]
    cmd.theta = target[2]
    publisher.publish(cmd)

def transport_load(publisher, src, dest):
    pos_idx = publisher_to_pos_idx[publisher]
    # load
    if pos[pos_idx] == 'start':
        if publisher == publisher_jackal1:
            move_ugv(publisher, start_front_jackal1)
        elif publisher == publisher_jackal2:
            move_ugv(publisher, start_front_jackal2)
        elif publisher == publisher_husky1:
            move_ugv(publisher, start_front_husky1)
    if src == 'AC1':
        if pos[pos_idx] == 'AC1' or pos[pos_idx] == 'start':
            move_ugv(publisher, L1_enter_U1)
        elif pos[pos_idx] == 'AC2':
            move_ugv(publisher, L1_enter_U2)
        elif pos[pos_idx] == 'AC3':
            move_ugv(publisher, L1_enter_U3)
        move_ugv(publisher, AC1_load)
    elif src == 'AC2':
        if pos[pos_idx] == 'AC1' or pos[pos_idx] == 'start':
            move_ugv(publisher, L2_enter_U1)
        elif pos[pos_idx] == 'AC2':
            move_ugv(publisher, L2_enter_U2)
        elif pos[pos_idx] == 'AC3':
            move_ugv(publisher, L2_enter_U3)
        move_ugv(publisher, AC2_load)
    elif src == 'AC3':
        if pos[pos_idx] == 'AC1' or pos[pos_idx] == 'start':
            move_ugv(publisher, L3_enter_U1)
        elif pos[pos_idx] == 'AC2':
            move_ugv(publisher, L3_enter_U2)
        elif pos[pos_idx] == 'AC3':
            move_ugv(publisher, L3_enter_U3)
        move_ugv(publisher, AC3_load)
    
def transport_load_exit(publisher, src, dest):
    pos_idx = publisher_to_pos_idx[publisher]
    # exit load
    if src == 'AC1':
        move_ugv(publisher, L1_exit)
        if dest == 'AC1':
            move_ugv(publisher, U1_enter1_L1)
        elif dest == 'AC2':
            move_ugv(publisher, U2_enter1_L1)
        elif dest == 'AC3':
            move_ugv(publisher, U3_enter1_L1)
    elif src == 'AC2':
        move_ugv(publisher, L2_exit)
        if dest == 'AC1':
            move_ugv(publisher, U1_enter1_L2)
        elif dest == 'AC2':
            move_ugv(publisher, U2_enter1_L2)
        elif dest == 'AC3':
            move_ugv(publisher, U3_enter1_L2)
    elif src == 'AC3':
        move_ugv(publisher, L3_exit)
        if dest == 'AC1':
            move_ugv(publisher, U1_enter1_L3)
        elif dest == 'AC2':
            move_ugv(publisher, U2_enter1_L3)
        elif dest == 'AC3':
            move_ugv(publisher, U3_enter1_L3)
    
def transport_unload(publisher, src, dest):
    pos_idx = publisher_to_pos_idx[publisher]
    # unload
    if dest == 'AC1':
        move_ugv(publisher, U1_enter2)
        move_ugv(publisher, AC1_unload)
    elif dest == 'AC2':
        move_ugv(publisher, U2_enter2)
        move_ugv(publisher, AC2_unload)
    elif dest == 'AC3':
        move_ugv(publisher, U3_enter2)
        move_ugv(publisher, AC3_unload)
    
     # exit unload

def transport_unload_exit(publisher, src, dest):
    pos_idx = publisher_to_pos_idx[publisher]
    # exit unload
    if dest == 'AC1':
        move_ugv(publisher, U1_exit)
        pos[pos_idx] = 'AC1'
    elif dest == 'AC2':
        move_ugv(publisher, U2_exit)
        pos[pos_idx] = 'AC2'
    elif dest == 'AC3':
        move_ugv(publisher, U3_exit)
        pos[pos_idx] = 'AC3'

def move_idle(publisher):
    pos_idx = publisher_to_pos_idx[publisher]
    if publisher == publisher_jackal1:
        move_ugv(publisher, start_jackal1)
    elif publisher == publisher_jackal2:
        move_ugv(publisher, start_jackal2)
    elif publisher == publisher_husky1:
        move_ugv(publisher, start_husky1)
    pos[pos_idx] = 'start'

follow_dict = dict()
obj_name = ['cardboard_box_L_1', 'cardboard_box_S_1', 'cardboard_box_M_1', 'cardboard_box_S_2']

def wait_transport(robot_list):
    global status_jackal1, status_jackal2, status_husky1
    rospy.sleep(0.1)
    # follow_dict = dict(robot_list)
    while status_jackal1+status_jackal2+status_husky1 > 0 and not rospy.is_shutdown():
        # for box_name, robot_name in follow_dict.items():
        #     robot_pose = get_model_coordinates(robot_name, '').pose
        #     box_state = ModelState()
        #     box_state.model_name = box_name
        #     box_state.pose.position.x = robot_pose.position.x
        #     box_state.pose.position.y = robot_pose.position.y
        #     box_state.pose.position.z = robot_pose.position.z + (0.33 if robot_name[:6] == 'jackal' else 0.4)
        #     box_state.pose.orientation.x = robot_pose.orientation.x
        #     box_state.pose.orientation.y = robot_pose.orientation.y
        #     box_state.pose.orientation.z = robot_pose.orientation.z
        #     box_state.pose.orientation.w = robot_pose.orientation.w

        #     set_model_coordinates(box_state)

        rate.sleep()

station_pos = [
    [(-1.4,-4.0),(-0.7,-4.0)],
    [(-1.4,-0.6),(-0.7,-0.6)],
    [(-1.4,2.5),(-0.7,2.5)],
]

def move_box_station(box_name, AC, station):
    box_state = ModelState()
    box_state.model_name = box_name
    box_state.pose.position.x = station_pos[AC][station][0]
    box_state.pose.position.y = station_pos[AC][station][1]
    box_state.pose.position.z = 0.6
    box_state.pose.orientation.x = 0
    box_state.pose.orientation.y = 0
    box_state.pose.orientation.z = 0
    box_state.pose.orientation.w = 1

    set_model_coordinates(box_state)

def move_box_exit(box_name, idx):
    box_state = ModelState()
    box_state.model_name = box_name
    box_state.pose.position.x = 4+idx
    box_state.pose.position.y = 9.5
    box_state.pose.position.z = 0.2
    box_state.pose.orientation.x = 0
    box_state.pose.orientation.y = 0
    box_state.pose.orientation.z = 0
    box_state.pose.orientation.w = 1

    set_model_coordinates(box_state)

def move_box_robot(robot_list, offset=False):
    follow_dict = dict(robot_list)
    for box_name, robot_name in follow_dict.items():
        robot_pose = get_model_coordinates(robot_name, '').pose
        box_state = ModelState()
        box_state.model_name = box_name
        box_state.pose.position.x = robot_pose.position.x
        box_state.pose.position.y = robot_pose.position.y
        box_state.pose.position.z = robot_pose.position.z + (0.33 if robot_name[:6] == 'jackal' else 0.4)
        box_state.pose.orientation.x = robot_pose.orientation.x
        box_state.pose.orientation.y = robot_pose.orientation.y
        box_state.pose.orientation.z = robot_pose.orientation.z
        box_state.pose.orientation.w = robot_pose.orientation.w

        if offset:
            box_state.pose.position.x = robot_pose.position.x - 0.1
            box_state.pose.position.y = robot_pose.position.y - 0.35
            box_state.pose.position.z = robot_pose.position.z + (0.33 if robot_name[:6] == 'jackal' else 0.4)
            box_state.pose.orientation.x = 0
            box_state.pose.orientation.y = 0
            box_state.pose.orientation.z = pi/2
            box_state.pose.orientation.w = pi/2

        set_model_coordinates(box_state)

# %%
def gazeboStateCallback(message):
    global publisher_gazebo_model_states, attach_list, do_attach

    last_message = message
    
    for attach in attach_list:
        object_name = attach[0]
        base_name = attach[1]
        offset = attach[2]
        follow = attach[3]

        try:
            base_idx = last_message.name.index(base_name)
            base_pose = last_message.pose[base_idx]

            new_object_state = ModelState()
            new_object_state.model_name = object_name
            new_object_state.pose.position.x = base_pose.position.x + offset[0]
            new_object_state.pose.position.y = base_pose.position.y + offset[1]
            new_object_state.pose.position.z = base_pose.position.z + offset[2]
            if follow:
                new_object_state.pose.orientation.x = base_pose.orientation.x
                new_object_state.pose.orientation.y = base_pose.orientation.y
                new_object_state.pose.orientation.z = base_pose.orientation.z
                new_object_state.pose.orientation.w = base_pose.orientation.w
            else:
                new_object_state.pose.orientation.x = 0
                new_object_state.pose.orientation.y = 0
                new_object_state.pose.orientation.z = 0
                new_object_state.pose.orientation.w = 1

            publisher_gazebo_model_states.publish(new_object_state)
        except:
            pass


# %%
gazebo_setmodelstate_topic = '/gazebo/set_model_state'
gazebo_linkstates_topic = '/gazebo/link_states'

attach_list = [
    # ('cardboard_box_S_1','Husky_husky1::base',(0,0,0.41),True)
    # ('cardboard_box_S_1','panda_multiple_arms::middle_arm_link7',(0,0,-0.37),False)
]    

publisher_gazebo_model_states = rospy.Publisher(gazebo_setmodelstate_topic, ModelState, queue_size=10)
subscriber_gazebo_link_states = rospy.Subscriber(gazebo_linkstates_topic, LinkStates, gazeboStateCallback)


# %%
# transport_load(publisher_jackal2, 'AC2', 'AC3')

# arm_load(group_middle)

# transport_load_exit(publisher_jackal2, 'AC2', 'AC3')
# transport_unload(publisher_jackal2, 'AC2', 'AC3')

# arm_unload(group_left)

# %%
# T=1
rospy.sleep(0.5)
move_arm(group_right, default, wait=True)
move_arm(group_middle, default, wait=True)
move_arm(group_left, default, wait=True)

arm_process(group_right, process=2)
arm_process(group_middle)
arm_process(group_left)

# %%
# T=2
transport_load(publisher_husky1, 'AC1', 'AC3')
transport_load(publisher_jackal1, 'AC2', 'AC1')

# wait_transport(['jackal1', 'husky1'])

wait_transport([(obj_name[0],'Husky_husky1'),
    (obj_name[1],'jackal1')])

# %%
# attach_links('panda_multiple_arms','right_arm_link7','cardboard_box_L_1','link')
# arm_load_group([(group_right,2)])
# detach_links('panda_multiple_arms','right_arm_link7','cardboard_box_L_1','link')
# move_box_robot([(obj_name[0],'Husky_husky1')])
# attach_links('Husky_husky1','base_link','cardboard_box_L_1','link')


# %%
attach_list = [('cardboard_box_L_1','panda_multiple_arms::right_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_right,2)])
attach_list = [('cardboard_box_L_1','Husky_husky1::base_link',(0,0,0.41),True),('cardboard_box_S_1','panda_multiple_arms::middle_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_middle,1)])
attach_list = [('cardboard_box_L_1','Husky_husky1::base_link',(0,0,0.41),True),('cardboard_box_S_1','jackal1::base_link',(0,0,0.33),True)]

arm_process(group_right, process=2, wait=False)

# %%
transport_load_exit(publisher_husky1, 'AC1', 'AC3')
transport_load_exit(publisher_jackal1, 'AC2', 'AC1')
transport_unload(publisher_husky1, 'AC1', 'AC3')
transport_unload(publisher_jackal1, 'AC2', 'AC1')

wait_transport([(obj_name[0],'Husky_husky1'),
    (obj_name[1],'jackal1')])

# %%
arm_unload1(group_right, process=2)
attach_list = [('cardboard_box_S_1','panda_multiple_arms::right_arm_link7',(0,0,-0.37),False)]
arm_unload2(group_right, process=2)
attach_list = []
move_box_station(obj_name[1],0,1)

arm_unload1(group_left, process=2)
attach_list = [('cardboard_box_L_1','panda_multiple_arms::left_arm_link7',(0,0,-0.37),False)]
arm_unload2(group_left, process=2)
attach_list = []
move_box_station(obj_name[0],2,1)

# %%
transport_unload_exit(publisher_husky1, 'AC1', 'AC3')
transport_unload_exit(publisher_jackal1, 'AC2', 'AC1')
move_idle(publisher_jackal2)

# wait_transport(['jackal1', 'jackal2', 'husky1'])

wait_transport([])

# %%
# T=3
move_idle(publisher_jackal1)
transport_load(publisher_husky1, 'AC3', 'AC2')
transport_load(publisher_jackal2, 'AC1', 'AC2')

# wait_transport(['jackal1', 'jackal2', 'husky1'])

wait_transport([])

# %%
arm_load1(group_left, process=1)
attach_list = [('cardboard_box_M_1','panda_multiple_arms::left_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_left,1)])
attach_list = [('cardboard_box_M_1','Husky_husky1::base_link',(0,0,0.41),True),('cardboard_box_S_1','panda_multiple_arms::right_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_right,2)])
attach_list = [('cardboard_box_M_1','Husky_husky1::base_link',(0,0,0.41),True),('cardboard_box_S_1','jackal2::base_link',(0,0,0.33),True)]
arm_process(group_left, process=1, wait=False)

# %%
transport_load_exit(publisher_husky1, 'AC3', 'AC2')

# wait_transport(['husky1'])

wait_transport([(obj_name[2],'Husky_husky1'),
    (obj_name[1],'jackal2')])

# %%
transport_load_exit(publisher_jackal2, 'AC1', 'AC2')
transport_unload(publisher_husky1, 'AC3', 'AC2')

# wait_transport(['jackal2', 'husky1'])

wait_transport([(obj_name[2],'Husky_husky1'),
    (obj_name[1],'jackal2')])

# %%
arm_unload1(group_middle, process=1)
attach_list = [('cardboard_box_M_1','panda_multiple_arms::middle_arm_link7',(0,0,-0.37),False),('cardboard_box_S_1','jackal2::base_link',(0,0,0.33),True)]
arm_unload2(group_middle, process=1)
attach_list = [('cardboard_box_S_1','jackal2::base_link',(0,0,0.33),True)]
move_box_station(obj_name[2],1,0)

# %%
transport_unload_exit(publisher_husky1, 'AC3', 'AC2')
transport_unload(publisher_jackal2, 'AC1', 'AC2')

# wait_transport(['jackal2', 'husky1'])

wait_transport([(obj_name[1],'jackal2')])

# %%
move_idle(publisher_husky1)

arm_unload1(group_middle, process=2)
attach_list = [('cardboard_box_S_1','panda_multiple_arms::middle_arm_link7',(0,0,-0.37),False)]
arm_unload2(group_middle, process=2)
attach_list = []
move_box_station(obj_name[1],1,1)

transport_unload_exit(publisher_jackal2, 'AC1', 'AC2')

# wait_transport(['jackal2', 'husky1'])

wait_transport([])

# %%
# T=4
move_idle(publisher_jackal1)

# move_idle(publisher_jackal2)
move_ugv(publisher_jackal2,AC3_load_team)

transport_load(publisher_husky1, 'AC3', 'AC1')
arm_process(group_middle, process=1, wait=True)

move_box_exit(obj_name[1],0)

# wait_transport(['jackal2', 'jackal1', 'husky1'])

wait_transport([])

# %%
attach_list = [('cardboard_box_L_1','panda_multiple_arms::left_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_left,1)])
# attach_list = [('cardboard_box_L_1','Husky_husky1::base_link',(0,0,0.41),True)]
attach_list = []


# %%
husky_state = ModelState()
husky_state.model_name = 'Husky_husky1'
husky_state.pose.position.x = -0.392678
husky_state.pose.position.y = 1.207838
husky_state.pose.position.z = 0.166497
husky_state.pose.orientation.x = 0
husky_state.pose.orientation.y = 0
husky_state.pose.orientation.z = -1
husky_state.pose.orientation.w = 0

set_model_coordinates(husky_state)

jackal_state = ModelState()
jackal_state.model_name = 'jackal2'
jackal_state.pose.position.x = -0.392678
jackal_state.pose.position.y = 0.453263
jackal_state.pose.position.z = 0.1   #0.097722
jackal_state.pose.orientation.x = 0
jackal_state.pose.orientation.y = 0
jackal_state.pose.orientation.z = -1
jackal_state.pose.orientation.w = 0

set_model_coordinates(jackal_state)

box_state = ModelState()
box_state.model_name = 'cardboard_box_L_1'
box_state.pose.position.x = -0.426048
box_state.pose.position.y = 0.758649
box_state.pose.position.z = 0.52   #0.516638
box_state.pose.orientation.x = -0.0027802
box_state.pose.orientation.y = 0.1743706
box_state.pose.orientation.z = -0.7024392
box_state.pose.orientation.w = 0.690048

set_model_coordinates(box_state)

# %%
attach_links('Husky_husky1','base_link','jackal2','base_link')
attach_links('Husky_husky1','base_link','cardboard_box_L_1','link')

# %%
transport_load_exit(publisher_husky1, 'AC3', 'AC1', )

# wait_transport(['husky1'])

# wait_transport([(obj_name[0],'Husky_husky1')])

# %%
transport_unload(publisher_husky1, 'AC3', 'AC1')

# wait_transport(['husky1'])

wait_transport([(obj_name[0],'Husky_husky1')])

# %%
detach_links('Husky_husky1','base_link','jackal2','base_link')
detach_links('Husky_husky1','base_link','cardboard_box_L_1','link')

# %%
arm_unload1(group_right, process=1)
attach_list = [('cardboard_box_L_1','panda_multiple_arms::right_arm_link7',(0,0,-0.37),False)]
arm_unload2(group_right, process=1)
attach_list = []
transport_unload_exit(publisher_husky1, 'AC3', 'AC1')

# wait_transport(['husky1'])

wait_transport([])

# %%
# T=5
transport_load(publisher_husky1, 'AC2', 'AC1')
arm_process(group_right, process=1)
move_box_exit(obj_name[3],1)

# wait_transport(['husky1'])

wait_transport([])

# %%
attach_list = [('cardboard_box_M_1','panda_multiple_arms::middle_arm_link7',(0,0,-0.37),False)]
arm_load_group([(group_middle, 1)])
attach_list = [('cardboard_box_M_1','Husky_husky1::base_link',(0,0,0.41),True)]

transport_load_exit(publisher_husky1, 'AC2', 'AC1')

# wait_transport(['husky1'])

wait_transport([(obj_name[2],'Husky_husky1')])

# %%
transport_unload(publisher_husky1, 'AC2', 'AC1')

# wait_transport(['husky1'])

wait_transport([(obj_name[2],'Husky_husky1')])

# %%
arm_unload1(group_right, process=2)
attach_list = [('cardboard_box_M_1','panda_multiple_arms::right_arm_link7',(0,0,-0.37),False)]
arm_unload2(group_right, process=2)
attach_list = []

move_box_station(obj_name[2],0,1)

transport_unload_exit(publisher_husky1, 'AC2', 'AC1')

# wait_transport(['husky1'])

wait_transport([])

# %%
# T=6
move_ugv(publisher_husky1, U3_exit)
move_idle(publisher_husky1)
arm_process(group_right, process=2)

# wait_transport(['husky1'])

wait_transport([])

# %%
# T=0 (Reset)

move_idle(publisher_jackal1)
move_idle(publisher_jackal2)
move_idle(publisher_husky1)

move_arm(group_right, default, wait=True)
move_arm(group_middle, default, wait=True)
move_arm(group_left, default, wait=True)

# %%
# attach_list = []
# move_box_station(obj_name[0],0,1)
# move_box_station(obj_name[1],1,0)
# move_box_station(obj_name[2],2,0)
# move_box_station(obj_name[3],0,0)

# %%



