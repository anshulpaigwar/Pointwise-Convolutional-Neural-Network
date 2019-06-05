from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch
import json
import h5py

# from IPython.core.debugger import Tracer
# debug_here = Tracer()

import numpy as np
import sys

# from sensor_msgs.msg import PointCloud2
# import std_msgs.msg
# import sensor_msgs.point_cloud2 as pcl2

import pcl
from tools.pcl_helper import *


import json


class Modelnet40_PCL_Dataset(data.Dataset):
    def __init__(self, data_dir, npoints = 2048, train = True, transform = None):
        self.npoints = npoints
        self.data_dir = data_dir
        self.train = train
        # train files
        self.train_files_path = os.path.join(self.data_dir, 'train_files.txt')
        self.test_files_path = os.path.join(self.data_dir, 'test_files.txt')

        self.train_files_list = [line.rstrip() for line in open(self.train_files_path)]
        self.test_files_list = [line.rstrip() for line in open(self.test_files_path)]
        self.transform = transform
	    # loading train files
        if self.train:
            print('loading training data ')
            self.train_data = []
            self.train_labels = []
            for file_name in self.train_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                # print("here")
                # print(file_path)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                self.train_data.append(data)
                self.train_labels.append(labels)
            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels)
        else:
            print('loading test data ')
            self.test_data = []
            self.test_labels = []

            for file_name in self.test_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                self.test_data.append(data)
                self.test_labels.append(labels)
            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels)



    def __getitem__(self, index):
    	if self.train:
    	    points, label = self.train_data[index], self.train_labels[index]
    	else:
    	    points, label = self.test_data[index], self.test_labels[index]

        if self.transform:
            points = self.subsample_points(points, resolution = 0.2)
            print(len(points))

    	return points, label


    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def subsample_points(self, points, resolution = 0.1):
        # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        cloud = pcl.PointCloud()

        #create pcl from points
        cloud.from_array(points)

        voxel_filter = cloud.make_voxel_grid_filter()
        voxel_filter.set_leaf_size(resolution, resolution, resolution)
        filter_cloud = voxel_filter.filter().to_array()
        return filter_cloud




if __name__ == '__main__':
    print('test')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.rstrip('misc')
    # d = Modelnet40_PCL_Dataset(data_dir= dir_path + 'datasets/modelnet40_ply_hdf5_2048', train=True)
    d = Modelnet40_PCL_Dataset(data_dir= "/home/anshul/inria_thesis/datasets/modelnet40_ply_hdf5_2048", train=True)
    print(len(d))
    print(d[0])

    points, label = d[0]
    # debug_here()
    print(points)
    print(points.shape)
    print(points.dtype)
    print(label.shape)
    print(label.dtype)

    # d = Modelnet40_PCL_Dataset(data_dir = dir_path + 'datasets/modelnet40_ply_hdf5_2048', train=False)
    d = Modelnet40_PCL_Dataset(data_dir = "/home/anshul/inria_thesis/datasets/modelnet40_ply_hdf5_2048", train=False)
    print(len(d))
    points, label = d[0]
    print(points)
    print(points.shape)
    print(points.dtype)
    print(label.shape)
    print(label.dtype)
