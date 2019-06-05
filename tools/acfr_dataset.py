from __future__ import print_function, division
import os
import sys
import numpy as np
import h5py
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
import pcl
from pcl_helper import *
import ipdb as pdb


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))


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



class acfr_dataset(data.Dataset):
    def __init__(self, data_dir, num_points=2048, mode='train'):
        self.npoints = num_points
        files = list_files(data_dir,"csv")
        dict = {'4wd': [0,1],'car':[0,1],'ute':[0,1],'van':[0,1],'building':[1,4],'pedestrian':[2,1],'traffic_lights':[3,1],'traffic_sign':[3,1],'tree':[4,2],
        'trunk':[5,2], 'truck':[6,4],'pole':[7,2],'post': [7,2], 'pillar':[8,3], 'bus': [9,4]}
        self.data = []
        self.labels = []
        for f in files:
            cat = f.split(".")[0]
            if dict.has_key(cat):
                point_dir = os.path.join(data_dir,f)
                d = np.genfromtxt(point_dir, delimiter=',')
                d = d[:, [3, 4, 5]]
                choice = np.random.choice(len(d), self.npoints, replace=True)
                # resample
                point_set = d[choice, :]
                for i in range(dict.get(cat)[1]):
                    self.data.append(point_set)
                    self.labels.append(dict.get(cat)[0])
        self.data = torch.from_numpy(np.stack(self.data)).float()
        # self.labels = torch.from_numpy(np.stack(self.labels)).float()
        self.labels = torch.ByteTensor(np.stack(self.labels))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


    def __len__(self):
        # print(len(self.data))
        return len(self.data)





def get_data_loaders(data_dir):

    datasets = acfr_dataset(data_dir)
    data_len = len(datasets)
    print("Total Data size ", data_len)
    indices = list(range(data_len))

    shuffle = True
    random_seed = 20
    batch_size = 32

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("using cuda")
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 4
        pin_memory = True




    # split the main dataset into three parts test(20%), valid(20%) and train (70%)
    split_train = int(np.floor(0.7 * data_len))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[:split_train]
    valid_idx = indices[split_train:]

    print("Train Data size ",len(train_idx))
    print("Valid Data size ",len(valid_idx))


    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(
        datasets, batch_size= batch_size, sampler=train_sampler, # BS = 105
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets, batch_size= batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return train_loader, valid_loader








if __name__ == '__main__':
    train_loader, valid_loader =  get_data_loaders(data_dir='/home/anshul/inria_thesis/datasets/Acfr/sydney-urban-objects-dataset/objects/')

    # get some random training images
    # dataiter = iter(train_loader)
    # data, labels = dataiter.next()
    for batch_idx, (data, labels) in enumerate(train_loader):
        B = data.shape[0] # Batch size
        N = data.shape[1] # Num of points in PointCloud
        color = np.array([1,1,1])
        color = np.vstack([color]*N)
        print(labels[batch_idx])
        ros_pub(data[batch_idx], "/all_points", color)
        pdb.set_trace()

    # data = acfr_dataset(data_dir='/home/anshul/inria_thesis/datasets/Acfr/sydney-urban-objects-dataset/objects/', mode='train')
