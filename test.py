import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as Transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from networks import *
from data_loader_4 import CreateDataloader
from data_loader_3 import CreateDataloader_3

### Parameters ###
split_ratio = 0
batch_size = 1
cuda = True

def main(args):

    Net = None
    test_loader = None
    class_num = 0

    if args.channel == 4:
        class_num, test_loader, _ = CreateDataloader(args.rgb_dir, args.d_dir, batch_size, split_ratio)
        #Net = ResNet18_(4, class_num)
        print('------------------------------------')
        print('Input Channel Size: ', args.channel)
        print('RGB Data Directory: ', args.rgb_dir)
        print('D Data Directory: ', args.d_dir)
    else:
        class_num, test_loader, _ = CreateDataloader_3(args.d_dir, batch_size, split_ratio)
        #Net = ResNet18_(3, class_num)
        print('------------------------------------')
        print('Input Channel Size: ', args.channel)
        print('Data Directory: ', args.d_dir)
        print('RGB Data Directory: ', args.rgb_dir)
        print('D Data Directory: ', args.d_dir)

    print('Load checkpoint: ', args.checkpoint)
    Net = torch.load(args.checkpoint)
    if cuda:
        Net.cuda()
    else:
        Net.cpu()
    loss_function = nn.CrossEntropyLoss()
    print('------------------------------------')

    ### Testing ###
    correct = 0
    total = 0
    for (images, labels) in tqdm(test_loader):
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        else:
            images, labels = images.cpu(), labels.cpu()

        outputs = Net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss = loss_function(outputs, labels)
    
    val_accu = 100 * float(correct) / float(total)
    print('Testing Accuracy: %d %%' % (val_accu))
        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--channel', type=int,
        help='Input layer channel size', default=4)
    parser.add_argument('--rgb_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/test/RGB')
    parser.add_argument('--d_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/test/D')
    parser.add_argument('--checkpoint', type=str,
        help='Folder to save checkpoints.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
