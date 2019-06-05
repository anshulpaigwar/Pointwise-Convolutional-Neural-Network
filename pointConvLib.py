#!/usr/bin/env python

"""
@Author: Anshul Paigwar
@email: p.anshul6@gmail.com

For more information on python-pcl check following links:

Git Hub repository:
https://github.com/strawlab/python-pcl
Check the examples and tests folder for sample coordinates

API documentation:
http://nlesc.github.io/python-pcl/
documentation is incomplete there are more available funtions

Udacity Nanodegree perception exercises for practice
https://github.com/udacity/RoboND-Perception-Exercises

check the documentation for pcl_helper.py

For more information on developing custom funtions
for pytorch check following links
https://pytorch.org/tutorials/advanced/cpp_extension.html

https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

https://pytorch.org/docs/master/notes/extending.html
"""





from __future__ import print_function

import math
import numpy as np
import ipdb as pdb

import torch
# import torch.nn as nn

# Our module!
import sort_attributes



class PointConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input ,
                    point_cloud,
                    out_grid_attributes,
                    num_points_in_quad,
                    kernel_radius,
                    kernel_size,
                    channels):
        output = sort_attributes.forward(input,
                                        point_cloud,
                                        out_grid_attributes,
                                        num_points_in_quad,
                                        kernel_radius,
                                        kernel_size,
                                        channels)
        # ctx.save_for_backward(input)


        return output

    @staticmethod
    def backward(ctx, grad_output):
        # outputs = sort_attributes.backward(
        #     grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        # d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        # print("grad_output", grad_output.shape)
        # input, = ctx.saved_tensors
        # print("input shape", input.shape)
        grad_input = torch.sum(grad_output,1)
        # print("grad_input", grad_input.shape)

        # pdb.set_trace()
        return grad_input, None, None, None, None, None, None


class pointConv(torch.nn.Module):
    def __init__(self,kernel_radius,kernel_size,channels):
        super(pointConv, self).__init__()
        self.kernel_radius = kernel_radius
        self.kernel_size = kernel_size
        self.channels = channels


    def forward(self, in_attributes, point_cloud):
        N = len(point_cloud)
        out_grid_attributes = torch.zeros(N,self.kernel_size*8,self.channels, requires_grad=False).cuda()
        num_points_in_quad = torch.zeros(N,self.kernel_size*8, requires_grad=False).cuda()
        return PointConvFunction.apply(in_attributes, point_cloud, out_grid_attributes, num_points_in_quad,
                                        self.kernel_radius, self.kernel_size, self.channels)


if __name__ == '__main__':

    N = 2048
    kernel_radius = 0.5
    kernel_size = 1
    channels = 2
    in_attributes = torch.ones(N,channels).cuda()
    out_grid_attributes = torch.zeros(N,kernel_size*8,channels).cuda()
    num_points_in_quad = torch.zeros(N,kernel_size*8).cuda()
    point_cloud =  torch.rand(N,3).cuda()

    a = PointConvFunction.apply(in_attributes, point_cloud, out_grid_attributes, num_points_in_quad,
                                    kernel_radius, kernel_size, channels)
    print(a[0])
