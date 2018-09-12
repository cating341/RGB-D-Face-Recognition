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

from tensorboardX import SummaryWriter
from networks import *
from data_loader_4 import CreateDataloader
from data_loader_3 import CreateDataloader_3

### Parameters ###
split_ratio = 0.1
epochs = 200
batch_size = 1
lr = 0.01
momentum = 0.5

cuda = True
log_step_percentage = 10

#data_path = 'D:/Datasets/BU_small/RGB'
#data_path_d = 'D:/Datasets/BU_small/D'

def main(args):
    train_loader_ = None
    valid_loader_ = None 
    class_num = 0

    if args.channel == 4:
        class_num, train_loader_, valid_loader_ = CreateDataloader(args.rgb_dir, args.d_dir, batch_size, split_ratio)
        Net = ResNet18_(4, class_num)
    else:
        class_num, train_loader_, valid_loader_ = CreateDataloader_3(args.d_dir, batch_size, split_ratio)
        Net = ResNet18_(3, class_num)

    if cuda:
        Net.cuda()

    model = Net

    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.LogSoftmax()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
    log_interval = len(train_loader_) / log_step_percentage


    # Add tensorboard writer
    writer = SummaryWriter()

    ### Training loop ###

    print('Start!')          
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader_, 0):
            # get the inputs
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = Net(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            #print(outputs.size(), predicted.size())
            
            total += labels.size(0)
            correct += (predicted == labels).sum()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            # print statistics
            if batch_idx % 500 == 0:
                n_iter = epoch*len(train_loader_) + batch_idx

                accu = 100 * correct / total
                writer.add_scalar('data/training_accuracy', accu, n_iter)
                writer.add_scalar('data/loss', loss, n_iter)
                print(f'Train Epoch: {epoch} [{batch_idx*len(inputs)}/{len(train_loader_.dataset)}] \t Loss: {loss.item():.5f} \t Accuracy: {accu}%')

        if epoch % 1 == 0:
            correct = 0
            total = 0
            #for data in val_loader:
            for (images, labels) in valid_loader_:
                #images, labels = data
                #labels = labels.type(torch.LongTensor)
                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                outputs = Net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                loss = loss_function(outputs, labels)
            
            val_accu = 100 * correct / total
            print('Accuracy: %d %%' % (val_accu))
            writer.add_scalar('data/val_accuracy', val_accu, epoch)
            writer.add_scalar('data/loss', loss, epoch)
        
        if epoch % 50 == 0:
            torch.save(Net, f'Net_{epoch}.pkl')
            print(f'Save network with epoch {epoch}')

    print('Finished Training')

    torch.save(Net, 'Net_final.pkl')
    print('Save network successfully!')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rgb_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_small/RGB')
    parser.add_argument('--d_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_small/D')
    parser.add_argument('--channel', type=int,
        help='Input layer channel size', default=4)
    #parser.add_argument('--class_num', type=int,
    #    help='Input layer channel size', default=100)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
