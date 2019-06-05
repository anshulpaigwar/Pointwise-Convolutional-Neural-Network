#!/usr/bin/env python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
from torch.distributions import Normal
from modules import PointWiseConvolution,PointWiseConvolutionX


class PointWiseConvNET(nn.Module):
    """
    A pointwise Convolution network for for semantic segmentation and object
    recognition with 3D point clouds. At the core of this network is point-wise
    convolution, a new convolution operator
    that can be applied at each point of a point cloud.
    rather than discretising the 3D space in grids.

    References
    ----------
    - Minh et. al., https://arxiv.org/pdf/1712.05245.pdf

    Tensorflow implementation: https://github.com/scenenn/pointwise
    """


    def __init__(self,N, num_clases): #FIXME
        """
        Initialize the PointWiseConvNET model and its
        different components.

        TODO: Change the below parameters
        Args
        ----
        - g: number of points in pointcloud
        - k: number of patches to extract per glimpse. (Zoom Values)
        - c: number of channels in each image.
        """
        super(PointWiseConvNET, self).__init__()

        self.N = N
        self.num_clases = num_clases
        self.convPW1 = PointWiseConvolutionX(c_in = 1,c_out = 9,kernel_size = 1,radius = 0.2) #TODO: Add stride features
        self.convPW2 = PointWiseConvolutionX(c_in = 9,c_out = 9,kernel_size = 1,radius = 0.3)
        self.convPW3 = PointWiseConvolutionX(c_in = 9,c_out = 9,kernel_size = 2,radius = 0.4)
        self.convPW4 = PointWiseConvolutionX(c_in = 9,c_out = 9,kernel_size = 2,radius = 0.5)
        self.fc1 = nn.Linear(self.N*9*4, 512)
        self.fc2 = nn.Linear(512, self.num_clases)


    def forward(self, points_tensor, attribute_tensor):
        """
        Args
        ----
        - points_tensor: (B,N,3) pointcloud. points_tensor is passed to
          all the convolution layer/ it is used for nearest neighbour search query
        - attribute_tensor: (B,N,Channels)  point attributes such as XYZ coordinates,
          RGB color, and normalized coordinates w.r.t. the room space it belongs to,
          normals, or other high-dimensional features output from
          preceding convolutional layers.

        Returns
        -------
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        """


        out_feat = None
        for B,points in enumerate(points_tensor):

            attribute_tensor1 = F.selu(self.convPW1(points, attribute_tensor[B]))
            attribute_tensor2 = F.selu(self.convPW2(points, attribute_tensor1))
            attribute_tensor3 = F.selu(self.convPW3(points, attribute_tensor2))
            attribute_tensor4 = F.selu(self.convPW4(points, attribute_tensor3))
            feat = torch.cat((attribute_tensor1, attribute_tensor2, attribute_tensor3, attribute_tensor4),0)
            # feat = torch.cat((attribute_tensor1, attribute_tensor2,attribute_tensor3),0)
            # print("feat", feat.shape)
            feat = feat.view(-1,self.N*9*4)
            # print("feat", feat.shape)
            feat = F.dropout(F.selu(self.fc1(feat)),training=self.training)
            # print("feat", feat.shape)
            feat = F.selu(self.fc2(feat))# (B,10) QUESTION: selu or softmax

            if out_feat is None:
                out_feat = feat
            else:
                out_feat = torch.cat((out_feat,feat),0)
        # print("feat",out_feat.shape)
        return out_feat
