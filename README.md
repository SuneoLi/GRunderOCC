# Gesture Recognition under Optical Camera Communication

This is my bachelor's dissertation project for the B.Eng. degree in the major of Computer Science and Technology at Sichuan University.

## 1 ROS Meta Package

### 1.1 Introduction

The main function of package recognition_under_occ is to receive an OCC-interfered gesture image, reconstruct it to eliminate the interference, recognize the gesture, and finally publish the result; The other two packages are used for testing. 

### 1.2 Run

```shell
roscore

rosrun turtlesim turtlesim_node
rosrun mock_camera mock_camera_1.py  # use gesture 1 as an example
rosrun recognition_under_occ recognition_under_occ.py  # the core package
rosrun mock_controller mock_controller.py

rqt_image_view  # optional
```

### 1.3 Environment

- Ubuntu 18.04
- ROS Melodic
- Python 3.6

## 2 Qt Desktop App

### 2.1 Introduction

The code is in the QtDesktopApp branch (this branch also include the ROSMetaPkg's code).

### 2.2 Run

```shell
python main.py
```

### 2.3 Environment

- PyQt 5
- Python 3.6

## 3 Pytorch Modeling Code

### 3.1 Introduction

The code is in the PytorchModelingCode branch (this branch also include the ROSMetaPkg's code), and it is an improved method of CycleGAN for this specialized task.

### 3.2 Run

same to CycleGAN

### 3.3 Environment

same to CycleGAN
