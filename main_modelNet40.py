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

"""


from __future__ import print_function

# Ros imports:
# import rospy
import math
import sys

# from sensor_msgs.msg import PointCloud2
# import std_msgs.msg
# import sensor_msgs.point_cloud2 as pcl2


import argparse
import random
import os
import shutil
import time
import numpy as np
import ipdb as pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import matplotlib.pyplot as plt

from torchviz import make_dot
import ipdb as pdb

# import pcl
# from tools.pcl_helper import *

import tools.modelnet40_pcl_datasets as modelnet40_dset
import tools.utils as utils
from model import PointWiseConvNET

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used










parser = argparse.ArgumentParser()

# specify data and datapath
parser.add_argument('--dataset',  default='modelnet40_pcl', help='modelnet40_pcl | ?? ')
parser.add_argument('--data_dir', default="/home/anshul/inria_thesis/datasets/modelnet40_ply_hdf5_2048", help='path to dataset')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--epochs', default=90, type=int,
                    help='number of total epochs to run')
parser.add_argument('--num_glimpses', default=6, type=int,
                    help='number of total epochs to run')

parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')

args = parser.parse_args()











# TODO: change the description below
"""
Loading the dataset:
# In this post we experiment with the classic Modelnet40_PCL_Dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.
"""


# dataset
if args.dataset == 'modelnet40_pcl':
	train_dataset = modelnet40_dset.Modelnet40_PCL_Dataset(args.data_dir, npoints=2048, train=True, transform = False)
	valid_dataset = modelnet40_dset.Modelnet40_PCL_Dataset(args.data_dir, npoints=2048, train=False, transform = False)
else:
	print('not supported dataset, so exit')
	exit()

print('number of train samples is: ', len(train_dataset))
print('number of test samples is: ', len(valid_dataset))
print('finished loading data')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                 shuffle=True, num_workers= 1, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchSize,
                                 shuffle=True, num_workers= 1, pin_memory=True)














# TODO: change the description below
"""
Initialising the model:

# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
"""

model = PointWiseConvNET(N = 2048, num_clases = 40)
if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optimizer = optim.Adam(model.parameters(), lr = 0.01)

criterion = nn.CrossEntropyLoss().cuda()
# criterion = nn.NLLLoss()
# criterion = nn.MultiLabelSoftMarginLoss()





# pdb.set_trace()

def train(epoch, subsample_points = True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()


    end = time.time()
    for batch_idx, (data, labels) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)


        # print("batch_idx",batch_idx)
        B = data.shape[0] # Batch size

        # if subsample_points:
        #     rand_indices = torch.from_numpy(np.random.randint(2048, size=500))
        #     input_data = torch.index_select(data, 1, rand_indices)
        #
        # else:
        #     input_data = data
        #
        N = data.shape[1] # Num of points in PointCloud
        # input_data = input_data.numpy()
        # print(N)


        # attributes is something thats gonna pass from one layer to another along with the point co-ordinates
        attributes = torch.ones(B,N,1) #(B,N,inputchannel)
        # attributes = data #(B,N,inputchannel)

        # print("atrributes shape",attributes.size)
        if use_cuda:
            labels, attributes, data=  labels.cuda(), attributes.cuda(), data.cuda()

        labels, attributes =  Variable(labels), Variable(attributes)
        labels = labels.view(labels.size(0))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data, attributes) #output.size = (B,classes)
        # print("forward_pass_time", time.time() - end)

        loss = criterion(output,labels.long())


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(prec1[0], B)
        top5.update(prec5[0], B)


        # print("loss_time", time.time() - end)
        loss.backward()
        # print("backward time", time.time() - end)
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))






def validate(subsample_points = True):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, labels) in enumerate(valid_loader):

            B = data.shape[0] # Batch size
            N = data.shape[1] # Num of points in PointCloud


            # attributes is something thats gonna pass from one layer to another along with the point co-ordinates
            attributes = torch.ones(B,N,1) #(B,N,inputchannel)
            # attributes = data
            # print("atrributes shape",attributes.size)
            if use_cuda:
                labels, attributes, data=  labels.cuda(), attributes.cuda(), data.cuda()

            labels, attributes =  Variable(labels), Variable(attributes)
            labels = labels.view(labels.size(0))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = model(data, attributes) #output.size = (B,classes)
            # print("forward_pass_time", time.time() - end)

            loss = criterion(output,labels.long())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), B)
            top1.update(prec1[0], B)
            top5.update(prec5[0], B)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg



best_prec1 = 0

def main():
    global args, best_prec1
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    for epoch in range(args.epochs):

        # adjust_learning_rate(optimizer, epoch)
        train(epoch)

        # evaluate on validation set
        prec1 = validate()


        # if (prec1 < best_prec1):
        #     adjust_learning_rate2(optimizer)


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)





'''
Save the model for later
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def adjust_learning_rate2(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = float(args.lr) / 4.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TODO: Repair the accuracy function


def accuracy(output, target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.view(target.size(0)).long()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def cal_accuracy(output, target):
    target = target.view(target.size(0)).long()
    with torch.no_grad():
        batch_size = target.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == target).sum().item()
        return correct * 100.0 / batch_size




def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp




def visualize(data, title):
    input_tensor = data.cpu()
    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))
    # Plot the results side-by-side
    plt.imshow(in_grid)
    plt.title(title)
    plt.show()








def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')




if __name__ == '__main__':
    main()
