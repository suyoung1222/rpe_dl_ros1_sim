# Gazebo Simulator: Learning for Voluntary Waiting and Subteaming (LVWS)

![Gazebo Screenshot](docs/img/screenshot.png)

This repository contains the Gazebo Simulator code for the paper [Learning for Dynamic Subteaming and Voluntary Waiting in Heterogeneous Multi-Robot Collaborative Scheduling](https://hcrlab.gitlab.io/project/lvws/). 


### Installation

First, install ROS noetic dependencies: 
```bash
sudo apt install \
    ros-noetic-husky-desktop \
    ros-noetic-jackal-desktop \
    ros-noetic-franka-ros \
    ros-noetic-moveit \
    ros-noetic-eigenpy \
    ros-noetic-geographic-msgs
sudo apt install python3-catkin-tools
```

Next, download the repository and build in a catkin workspace: 
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/wdjose/lvws_demo.git
cd ~/catkin_ws
catkin build
```

### Execution

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch lvws_demo warehouse.launch
```

Because the panda_multiple_arms package is not very stable during initialization (ROS timing issues), sometimes not all the arms will move. You can reset the simulation and try again. 

To check if all components have been loaded correctly, you can compare the topc list at [topic_list.txt](docs/topic_list.txt) with your currently loaded topics (`rostopic list`). 
# rpe_dl_ros1_sim


### Pose Estimation

## Requirement
```bash
roscd husky_description/urdf/
vim husky.urdf.xacro
# modify below lines
<xacro:arg name="realsense_enabled"             default="$(optenv HUSKY_REALSENSE_ENABLED 1)" />
<xacro:arg name="realsense_xyz"                 default="$(optenv HUSKY_REALSENSE_XYZ 0 0 0.2)" />
```

terminal1:
```bash
roslaunch multihusky_gazebo multihusky_playpen.launch
```

terminal2:
```bash
roslaunch multihusky_gazebo multi_husky_teleop.launch 
```

terminal3:
```bash
rosrun multihusky_gazebo run_task_sim.py 
```


