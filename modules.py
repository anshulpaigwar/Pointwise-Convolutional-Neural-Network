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


This code contains the pytorch implementation of pointwise convolution operator.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
import math

# from tools.knn_cuda import knn

import ipdb as pdb

# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
import pcl
from tools.pcl_helper import *

from pointConvLib import pointConv

def ros_pub(points,topic,color):

    rospy.init_node('pcl2_pub_example', anonymous=True)
    rospy.loginfo("Initializing sample pcl2 publisher node...")
    pcl_pub = rospy.Publisher(topic, PointCloud2, queue_size=10)

    #give time to roscore to make the connections
    rospy.sleep(1.)
    points = points.cpu().detach().numpy()

    # color = np.array([1,0.5,0])
    # color = np.vstack([color]*N)
    cloud_msg = xyzrgb_array_to_pointcloud2(points,color,stamp =rospy.Time.now(), frame_id = "map" )
    rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)



def _out_size(input_size, kernel_size, stride = 1, padding = 0, pool = False,  pool_kernel_size = 2):
    out_size = (input_size - kernel_size + 2 * padding)/stride + 1
    # flat_features = output_size * output_size * channel
    if pool:
        out_size = out_size/pool_kernel_size
    return int(out_size)


# def generate_octree(points, resolution): #FIXME: Octree should be generated only once in the network not for every convolution operation
#     """
#     this funtion takes the point cloud of single object and returns an octree
#     """
#     # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
#     cloud = pcl.PointCloud()
#
#     # points = points.data.cpu()
#     # points = points.numpy() #FIXME: copying memory from gpu to cpu is not good
#     #create pcl from points
#     cloud.from_array(points)
#
#     # x,y,z Area Filter
#     # resolution = resolution
#     octree = pcl.OctreePointCloudSearch(resolution)
#     octree.set_input_cloud(cloud)
#
#     octree.define_bounding_box()
#
#     octree.add_points_from_input_cloud()
#
#     return octree






class STN3d(nn.Module):
    def __init__(self, c_in = 3, num_points = 2048):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(c_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x # transformation matrix

















class PointWiseConvolutionBatch(torch.nn.Module):
    """
    TODO: Change the description below:


    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, c_in = 1, c_out = 9, kernel_size = 2, radius = 0.1, knn = 50):
        super(PointWiseConvolutionBatch,self).__init__()
        self.knn = knn #K-nearest neighbour
        self.k = kernel_size
        self.num_kernel_cells = kernel_size * 8
        self.c_in = c_in
        self.c_out = c_out
        self.radius = radius
        self.conv1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=self.num_kernel_cells, padding = 0)

    def forward(self, points_tensor, batch_atributes):
        # points_tensor.size = (B,2048,3)
        # batch_attributes.size = (B,2048,c_in)
        print(batch_atributes.shape)
        # xyz = batch_atributes.clone()
        # b = points_tensor.shape[0]
        # p = points_tensor.shape[1]

        # here we create an another variable to store the attributes after convolution
        #out_atributes = batch_atributes
        # resize to change the number of channels from c_in to c_out
        #out_atributes.resize_(b,p,self.c_out,1)

        out_atributes = None

        for B,points in enumerate(points_tensor):
            # points.size = (2048, 3)

            # IDEA: is that grid values is constructed by combination of attributes so should be auto differential
            grid_attributes = torch.zeros(len(points), self.num_kernel_cells, self.c_in)
            grid_attributes = grid_attributes.cuda()
            grid_attributes = Variable(grid_attributes)
            grid = torch.zeros(2048,2048).cuda()

            output  = sort_in_grid.forward(points,grid,self.radius,self.k, self.c_in).cuda() #TODO: release the memory of grid
            output = output.permute(1,0) # TODO make permute in cuda file


            for i in range(len(points)):
                for quad in range(self.num_kernel_cells):
                    # pdb.set_trace()
                    ind = torch.squeeze((output[i]== quad).nonzero())
                    points_count_in_quad = ind.nelement()
                    if(points_count_in_quad != 0):
                        # print(len(ind))
                        a = torch.index_select(batch_atributes[B], 0,ind)
                        grid_attributes[i,quad] = torch.sum(a, 0)/points_count_in_quad



            grid_attributes = grid_attributes.permute(0,2,1) # grid_attributes = (2048,channels,quad)
            # Convolution here
            temp = self.conv1(grid_attributes) # temp.size = (2048,c_out,1)
            temp = temp.squeeze(2)
            # print(temp.shape)
            temp = temp.unsqueeze(0)
            #print("yeah")
            if out_atributes is None:
                out_atributes = temp
            else:
                out_atributes = torch.cat((out_atributes,temp),0)
        return out_atributes




class PointWiseConvolution(torch.nn.Module):
    """
    TODO: Change the description below:


    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, c_in = 1, c_out = 9, kernel_size = 2, radius = 0.1, knn = 50): # TODO remove knn
        super(PointWiseConvolution,self).__init__()
        self.knn = knn #K-nearest neighbour
        self.k = kernel_size
        self.num_kernel_cells = kernel_size * 8
        self.c_in = c_in
        self.c_out = c_out
        self.radius = radius
        self.conv1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=self.num_kernel_cells, padding = 0)

    def forward(self, points, attributes):

        # N = len(points)
        # color = np.array([1,1,1])
        # color = np.vstack([color]*N)
        # ros_pub(points, "/all_points", color)

        # attributes.size = (2048,c_in)
        # points.size = (2048, 3)

        # IDEA: is that grid values is constructed by combination of attributes so should be auto differential
        grid_attributes = torch.zeros(len(points), self.num_kernel_cells, self.c_in)
        grid_attributes = grid_attributes.cuda()
        grid_attributes = Variable(grid_attributes)
        output = torch.zeros(2048,2048).cuda()

        output  = sort_in_grid.forward(points, output, self.radius, self.k).cuda() #TODO: release the memory of grid
        output = output.permute(1,0) # TODO make permute in cuda file


        for i in range(len(points)):


            # ind = torch.squeeze((output[i]!= -2).nonzero())
            # a = torch.index_select(points, 0, ind)
            #
            # N = len(a)
            # color = np.array([1,0.5,0])
            # color = np.vstack([color]*N)
            # ros_pub(a, "/kernel", color)

            kernel_points = None
            kernel_color = None

            for quad in range(self.num_kernel_cells):
                # pdb.set_trace()
                ind = torch.squeeze((output[i]== quad).nonzero())
                points_count_in_quad = ind.nelement()
                if(points_count_in_quad != 0):
                    a = torch.index_select(attributes, 0,ind)
                    grid_attributes[i,quad] = torch.sum(a, 0)/points_count_in_quad

            #         p = torch.index_select(points, 0, ind)
            #         N = len(p)
            #         color = np.array([0.1 * quad, 1 / (quad+0.001), 0])
            #         color = np.vstack([color]*N)
            #
            #         if kernel_points is None:
            #             kernel_points = p
            #             kernel_color = color
            #         else:
            #             kernel_points = torch.cat((kernel_points,p),0)
            #             kernel_color = np.vstack([kernel_color,color])
            # ros_pub(kernel_points, "/kernel", kernel_color)





        # pdb.set_trace()
        grid_attributes = grid_attributes.permute(0,2,1) # grid_attributes = (2048,channels,quad)
        # Convolution here
        temp = self.conv1(grid_attributes) # temp.size = (2048,c_out,1)
        out_atributes = temp.squeeze(2) # out_atributes = (2048,c_out)
        # out_atributes = temp.unsqueeze(0)
        # print(out_atributes.shape)
        return out_atributes




class PointWiseConvolutionX(torch.nn.Module):
    """
    TODO: Change the description below:


    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, c_in = 1, c_out = 9, kernel_size = 2, radius = 0.1): # TODO remove knn
        super(PointWiseConvolutionX,self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel_cells = kernel_size * 8
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_radius = radius
        self.conv1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=self.num_kernel_cells, padding = 0)
        self.pconv = pointConv(self.kernel_radius,self.kernel_size,self.c_in)

    def forward(self, points, attributes):

        # N = len(points)
        # color = np.array([1,1,1])
        # color = np.vstack([color]*N)
        # ros_pub(points, "/all_points", color)

        # attributes.size = (2048,c_in)
        # points.size = (2048, 3)

        '''
        Important the inputs should be of type float not double not float64
        tensor should be contiguous
        '''
        grid_attributes  = self.pconv(attributes,points)

        # sorted_points = torch.zeros(2048,2048).cuda()
        # sorted_points  = sort_attributes.output_sorted_points(points, sorted_points, self.kernel_radius, self.kernel_size).cuda() #TODO: release the memory of grid
        # sorted_points = output.permute(1,0) # TODO make permute in cuda file
        #
        #
        # for i in range(len(points)):
        #     kernel_points = None
        #     kernel_color = None
        #
        #     for quad in range(self.num_kernel_cells):
        #         # pdb.set_trace()
        #         ind = torch.squeeze((output[i]== quad).nonzero())
        #         points_count_in_quad = ind.nelement()
        #         if(points_count_in_quad != 0):
        #             a = torch.index_select(attributes, 0,ind)
        #             grid_attributes[i,quad] = torch.sum(a, 0)/points_count_in_quad
        #
        #     #         p = torch.index_select(points, 0, ind)
        #     #         N = len(p)
        #     #         color = np.array([0.1 * quad, 1 / (quad+0.001), 0])
        #     #         color = np.vstack([color]*N)
        #     #
        #     #         if kernel_points is None:
        #     #             kernel_points = p
        #     #             kernel_color = color
        #     #         else:
        #     #             kernel_points = torch.cat((kernel_points,p),0)
        #     #             kernel_color = np.vstack([kernel_color,color])
        #     # ros_pub(kernel_points, "/kernel", kernel_color)
        #




        # pdb.set_trace()
        grid_attributes = grid_attributes.permute(0,2,1) # grid_attributes = (2048,channels,quad)
        # Convolution here
        temp = self.conv1(grid_attributes) # temp.size = (2048,c_out,1)
        out_atributes = temp.squeeze(2) # out_atributes = (2048,c_out)
        # out_atributes = temp.unsqueeze(0)
        # print(out_atributes.shape)
        return out_atributes



        #
        #
        #
        # # IDEA: is that grid values is constructed by combination of attributes so should be auto differential
        # grid_attributes = torch.zeros(len(points), self.num_kernel_cells, self.c_in)
        # grid_attributes = grid_attributes.cuda()
        # grid_attributes = Variable(grid_attributes)
        #
        #





                    # print(b.shape)
                    # print(b)
                    # pdb.set_trace()
                    #
                    # grid_attributes[i,quad] = torch.sum(torch.index_select(batch_atributes,1,ind), 0)/points_count_in_quad

                #     quad_attribute = batch_atributes[b]
                #     quadrant = torch.zeros(self.num_kernel_cells, self.c_in).cuda()
                # good_point = points[i]
                # quadrant = torch.zeros(self.num_kernel_cells, self.c_in).cuda()
                # quadrant = Variable(quadrant)
                #
                #
                # points_count_in_quad = torch.zeros(self.num_kernel_cells,1).cuda()
                # points_count_in_quad = Variable(points_count_in_quad)
                #
                #
                # for counter, j in enumerate(indices[i]):
                #     j = j-1 # as j is starting from 1 to 2048
                #     neighbor_point = points[j]
                #     mag = dist[i,counter]
                #     if mag > 0:
                #         if (mag < self.radius):
                #             quad = which_cell(neighbor_point - good_point, self.radius, self.k, mag) #TODO: change quadrant funtion for kernel size 16
                #             # grid_attributes[i,quad] = grid_attributes[i,quad] + batch_atributes[B,j]
                #             quadrant[quad] = quadrant[quad] + batch_atributes[B,j]
                #             points_count_in_quad[quad] = points_count_in_quad[quad] + 1
                # points_count_in_quad[points_count_in_quad == 0] =1 #COMBAK
                # grid_attributes[i] = quadrant / points_count_in_quad









            # dist, indices = knn.knn(np.transpose(points),np.transpose(points),self.knn)
            # # indices = indices.permute(1,0)
            # indices = np.transpose(indices)
            # dist = np.transpose(dist)

            # print("indices", indices.shape) # indices.size = (2048,self.knn)
            # print("dist", dist) # indices.size = (2048,self.knn)











# class PointWiseConvolutionBatch(torch.nn.Module):
#     """
#     TODO: Change the description below:
#
#
#     A network that combines the "what" and the "where"
#     into a glimpse feature vector `g_t`.
#
#     In other words:
#
#         `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`
#
#     Args
#     ----
#     - h_g: hidden layer size of the fc layer for `phi`.
#     - h_l: hidden layer size of the fc layer for `l`.
#
#     Returns
#     -------
#     - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
#       representation returned by the glimpse network for the
#       current timestep `t`.
#     """
#     def __init__(self, c_in = 1, c_out = 9, kernel_size = 2, radius = 0.1):
#         super(PointWiseConvolutionBatch,self).__init__()
#         self.k = kernel_size
#         self.num_kernel_cells = kernel_size * 8
#         self.c_in = c_in
#         self.c_out = c_out
#         self.radius = radius
#         self.conv1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=self.num_kernel_cells, padding = 0)
#
#     def forward(self, points_array, batch_atributes):
#         # points_tensor.size = (B,2048,3)
#         # batch_attributes.size = (B,2048,c_in,1)
#         print(batch_atributes.shape)
#         b = points_array.shape[0]
#         p = points_array.shape[1]
#
#         # here we create an another variable to store the attributes after convolution
#         #out_atributes = batch_atributes
#         # resize to change the number of channels from c_in to c_out
#         #out_atributes.resize_(b,p,self.c_out,1)
#
#         out_atributes = None
#
#         for B,points in enumerate(points_array):
#             # points.size = (2048, 3)
#
#             # IDEA: is that grid values is constructed by combination of attributes so should be auto differential
#             grid_attributes = torch.zeros(len(points), self.c_in, self.num_kernel_cells)
#             grid_attributes = grid_attributes.cuda()
#             grid_attributes = Variable(grid_attributes)
#
#
#             # TODO: convert the point variable to numpy  to generate octree
#             octree = generate_octree( points, resolution = 0.1)
#
#             for i in range(len(points)):
#                 print(i)
#                 good_point = points[i]
#                 indices,sqr_dist = octree.radius_search(good_point, self.radius)
#
#                 for j in indices:
#                     neighbor_point = points[j]
#                     quad = which_cell(neighbor_point - good_point, self.radius, self.k) #TODO: change quadrant funtion for kernel size 16
#
#                     for channel in range(self.c_in):
#                         #print(B,j,channel)
#                         temp_value = grid_attributes[i,channel,quad] + batch_atributes[B,j,channel,0] #COMBAK: Normalise the values in each quadrant
#                         grid_attributes[i,channel,quad] = temp_value
#             # Convolution here
#             temp = self.conv1(grid_attributes) # temp.size = (2048,c_out,1)
#             temp = temp.unsqueeze(0)
#             #print("yeah")
#             if out_atributes is None:
#                 out_atributes = temp
#             else:
#                 out_atributes = torch.cat((out_atributes,temp),0)
#         return out_atributes









# def which_cell(vec, radius, k, mag):
#     x,y,z = vec
#     x,y,z = int(x),int(y),int(z)
#     x = 0 if x <= 0 else 1
#     y = 0 if y <= 0 else 2
#     # mag = math.sqrt(np.dot(vec, vec))
#
#     # print(mag)
#     vox_radius = radius/k
#     if (z >= 0):
#         cell_num = x + y
#     else:
#         cell_num = x + y + 4
#
#     cell_num = cell_num + (mag//vox_radius)*8
#     return int(cell_num)
#
#

# class KNearestNeighbor(Function):
#   """Accumulate x += y using broadcasting sum.
#   """
#   def __init__(self, k):
#     self.k = k
#
#   def forward(self, ref, query):
#     ref = ref.float().cuda()
#     query = query.float().cuda()
#
#     inds = torch.zeros(self.k, query.shape[1]).long().cuda()
#     dists = torch.zeros(self.k, query.shape[1]).float().cuda()
#
#     knn_pytorch.knn(ref, query, inds, dists)
#
#     return inds, dists
#
#


# TODO: Change the python code first
# TODO: Write your own Cuda Code
