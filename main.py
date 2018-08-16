import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as Transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from tensorboardX import SummaryWriter
from networks import *
from data_loaders import CreateDataloader

### Parameters ###
split_ratio = 0.1
epochs = 200
batch_size = 1
lr = 0.01
momentum = 0.5

cuda = True
log_step_percentage = 10
'''
class DataType():
    "Stores the paths to images for a given class"
    def __init__(self, name, rgb_paths, d_paths):
        self.class_name = name
        self.rgb_paths = rgb_paths
        self.d_paths = d_paths

class ImagePath():
    def __init__(self, rgb_path, d_path):
        self.rgb_path = rgb_path
        self.d_path = d_path

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]

    return image_paths

def get_labels(data):
    images = []
    labels = []
    for i in range(len(data)):
        images += [ImagePath(data[i].rgb_paths[j], data[i].d_paths[j]) for j in range(len(data[i].rgb_paths))]
        labels += [i] * len(data[i].rgb_paths)
    
    return np.array(images), np.array(labels)

data_path = 'D:/Datasets/BU_3DFE/RGB'
data_path_d = 'D:/Datasets/BU_3DFE/D'

dataset = []
classes = [path for path in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, path))]

classes_n = len(classes)

for i in range(classes_n):
    face_dir = os.path.join(data_path, classes[i])
    face_dir_d = os.path.join(data_path_d, classes[i])
    
    # Get image pathes of this class
    image_paths = get_image_paths(face_dir)
    image_paths_d = get_image_paths(face_dir_d)

    dataset.append(DataType(classes[i], image_paths, image_paths_d))


train_x, train_y = get_labels(dataset)

print('Number of classes: %d' % classes_n)
print('Total images: %d (split ratio: %.1f)' % (len(train_x), split_ratio) )

def my_loader(path, Type):
    #print(path)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if Type == 3:
                img = img.convert('RGB')
            elif Type == 1:
                img = img.convert('L')
            return img

class MyDataset(Data.Dataset):
    def __init__(self, img_paths, labels, transform, loader=my_loader):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index): #return data type is tensor
        rgb_path, d_path = self.img_paths[index].rgb_path, self.img_paths[index].d_path
        label = self.labels[index]
        
        rgb_img = np.array( my_loader(rgb_path, 3) ) 
        d_img = np.array( my_loader(d_path, 1) ) 
        d_img = np.expand_dims(d_img, axis=2)

        img = np.append(rgb_img, d_img, axis=2)
        img = self.transform(Image.fromarray(img))
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)

        
        return img, label

    def __len__(self): # return the total size of the dataset
        return len(self.labels)

transform = Transforms.Compose([
    Transforms.Resize(224),
    Transforms.ToTensor(),
])
    
dataset_t = MyDataset(train_x, train_y, transform=transform)


### Split dataset and creat train & valid dataloader ###

num_train = len(dataset_t)
indices = list(range(num_train))
split = int(np.floor(split_ratio * num_train))

#np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader_ = torch.utils.data.DataLoader(dataset_t, batch_size=batch, sampler=train_sampler)
valid_loader_ = torch.utils.data.DataLoader(dataset_t, batch_size=batch, sampler=valid_sampler)

print('Training images:', len(train_loader_))
print('Validation images: ', len(valid_loader_))
'''

data_path = 'D:/Datasets/BU_3DFE/RGB'
data_path_d = 'D:/Datasets/BU_3DFE/D'


train_loader_, valid_loader_ = CreateDataloader(data_path, data_path_d, batch_size, split_ratio)


Net = ResNet18_(4, 20)
if cuda:
    Net.cuda()

model = Net

loss_function = nn.CrossEntropyLoss()
#loss_function = nn.LogSoftmax()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
log_interval = len(train_loader_) / log_step_percentage


# Add tensorboard writer
writer = SummaryWriter()

### Training loop ###

print('Start!')          
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
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
        
        writer.add_scalar('data/loss', loss, epoch)

        # print statistics
        if batch_idx % 500 == 0:
            accu = 100 * correct / total
            writer.add_scalar('data/training_accuracy', accu, epoch)
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

        val_accu = 100 * correct / total
        writer.add_scalar('data/val_accuracy', val_accu, epoch)
        print('Accuracy: %d %%' % (val_accu))
    

print('Finished Training')

torch.save(Net, 'Net_all_BU.pkl')
print('Save network successfully!')