import torch
import torchvision
import torch.nn as nn
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
import networks
from data_loader_4 import CreateDataloader
from data_loader_3 import CreateDataloader_3




def main(args):
    ### Parameters ###
    split_ratio = args.split
    epochs = 500
    batch_size = 1
    lr = 0.01
    momentum = 0.5

    cuda = True
    log_step_percentage = 10
    checkpoint_path = './checkpoints'

    train_loader = None
    valid_loader = None 
    class_num = 0

    if args.channel == 4:
        class_num, train_loader, valid_loader = CreateDataloader(args.rgb_dir, args.d_dir, batch_size, split_ratio)
        if split_ratio == 0:
            _, valid_loader, _ = CreateDataloader(args.rgb_dir_test, args.d_dir_test, batch_size, split_ratio)
        Net = networks.ResNet18(4, class_num)
        print('------------------------------------')
        print('Input Channel Size: ', args.channel)
        print('RGB Data Directory: ', args.rgb_dir)
        print('D Data Directory: ', args.d_dir)
    else:
        class_num, train_loader, valid_loader = CreateDataloader_3(args.d_dir, batch_size, split_ratio)
        if split_ratio == 0:
            _, valid_loader, _ = CreateDataloader_3(args.d_dir_test, batch_size, split_ratio)
        Net = nwtworks.ResNet18(3, class_num)
        print('------------------------------------')
        print('Input Channel Size: ', args.channel)
        print('Data Directory: ', args.d_dir)
        print('RGB Data Directory: ', args.rgb_dir)
        print('D Data Directory: ', args.d_dir)

    # checkpints path
    #if not os.path.exists(f'{checkpoint_path}/{args.checkpoint_name}'):
    #os.makedirs(f'{checkpoint_path}/{args.checkpoint_name}')
    try:
        os.stat('%s/%s' % (checkpoint_path, args.checkpoint_name))
    except:
        os.mkdir('%s/%s' % (checkpoint_path, args.checkpoint_name))

    if cuda:
        Net.cuda()

    model = Net

    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.LogSoftmax()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
    log_interval = len(train_loader) / log_step_percentage


    # Add tensorboard writer
    writer = SummaryWriter()

    ### Training loop ###

    print('Start!')          
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = Net(inputs)
            #print(outputs)
            #import pdb
            #pdb.set_trace()
            _, predicted = torch.max(outputs, 1)
            
            #print(outputs.size(), predicted.size())
            
            total += labels.size(0)
            correct += (predicted == labels).sum()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            # print statistics
            if batch_idx % 10000 == 0 and batch_idx != 0:
                n_iter = epoch*len(train_loader) + batch_idx

                accu = 100 * float(correct) / float(total)
                writer.add_scalar('data/training_accuracy', accu, n_iter)
                writer.add_scalar('data/loss', loss.item(), n_iter)
                print(f'Train Epoch: {epoch} [{batch_idx*len(inputs)}/{len(train_loader.dataset)}] \t Loss: {loss.item()} \t Accuracy: {accu:.2f}%')

        if epoch % 1 == 0:
            correct = 0
            total = 0
            #for data in val_loader:
            for (images, labels) in valid_loader:
                #images, labels = data
                #labels = labels.type(torch.LongTensor)
                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                outputs = Net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                loss = loss_function(outputs, labels)
            
            val_accu = 100 * float(correct) / float(total)
            print('Validation Accuracy: %d %%' % (val_accu))
            writer.add_scalar('data/val_accuracy', val_accu, epoch)
            writer.add_scalar('data/loss', loss, epoch)
        
        if epoch == 50:
            lr = lr/2
        if epoch %  1 == 0:
            torch.save(Net, f'{checkpoint_path}/{args.checkpoint_name}/Net_{epoch}.pkl')
            print(f'Save network: {checkpoint_path}/{args.checkpoint_name}/Net_{epoch}.pkl')

    print('Finished Training')

    torch.save(Net, 'Net_final.pkl')
    print('Save network successfully!')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/train/RGB')
    parser.add_argument('--d_dir', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/train/D')
    parser.add_argument('--rgb_dir_test', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/test/RGB')
    parser.add_argument('--d_dir_test', type=str, 
        help='RGB dataset.', default='D:/Datasets/BU_1225/dataset/test/D')
    parser.add_argument('--channel', type=int,
        help='Input layer channel size', default=4)
    parser.add_argument('--checkpoint_name', type=str,
        help='Folder to save checkpoints.', default='tmp')
    parser.add_argument('--split', type=float,
        help='Split ratio', default=0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
