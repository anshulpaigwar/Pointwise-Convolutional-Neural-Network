## Introduction
This is unofficial PyTorch implementation of [Pointwise Convolutional Neural Networks](https://arxiv.org/pdf/1712.05245.pdf) in CVPR 2018.
Official project page for this work can be found [here](http://www.saikit.org/projects/sceneNN/home/cvpr18/index.html).

## Installation

We have tested the algorithm on the system with Ubuntu 16.04, 8 GB RAM
and NVIDIA GTX-1080.
### Dependencies
```
Python 2.7
CUDA 9.1
PyTorch 1.0
scipy
shapely
```
### Visualization
For visualization of the output bounding boxes and easy integration
with our real system we use the Robot Operating System (ROS):
```
ROS
PCL
```

## Implementation
<img src="https://github.com/anshulpaigwar/Pointwise-Convolutional-Neural-Network/blob/master/doc/pointwise_operator.png" alt="drawing" width="800"/>

## Training Data
<img src="https://github.com/anshulpaigwar/Pointwise-Convolutional-Neural-Network/blob/master/doc/3D_Dataset.png" alt="drawing" width="800"/>

## Usages

<img src="https://github.com/anshulpaigwar/Pointwise-Convolutional-Neural-Network/blob/master/doc/experiments.png" alt="drawing" width="800"/>
