# Gesture Recognition under Optical Camera Communication

This ROS metapackage is part of my bachelor's dissertation project for the B.Eng. degree in the major of Computer Science and Technology at Sichuan University.

## 1 Introduction

The main function of package recognition_under_occ is to receive an OCC-interfered gesture image, reconstruct it to eliminate the interference, recognize the gesture, and finally publish the result; The other two packages are used for testing. 

## 2 Testing

```shell
roscore

rosrun turtlesim turtlesim_node
rosrun mock_camera mock_camera_1.py  # use gesture 1 as an example
rosrun recognition_under_occ recognition_under_occ.py  # the core package
rosrun mock_controller mock_controller.py

rqt_image_view  # optional
```

## 3 Environment

- Ubuntu 18.04
- ROS Melodic
- Python 3.6.13