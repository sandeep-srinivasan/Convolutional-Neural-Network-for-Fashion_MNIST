#!/usr/bin/env python
# coding: utf-8

# # Comparison Systems

# In[1]:


# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # TensorBoard support


# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = False,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)


# In[ ]:


#System with different number of feature maps

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        self.fc1 = nn.Linear(in_features=4*4*64, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
        
    def forward(self, t):
        #implement the forward pass
        
        #For Convolution 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #For Convolution 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #For FC 1
        t = t.reshape(-1,4*4*64)
        t = self.fc1(t)
        t = F.relu(t)

        
        #For FC 2
        t = self.fc2(t)
        t = F.relu(t)
        
        #Output
        t = self.out(t)
        
        # Since cross-entropy is used as activation, there is no need for softmax
        return(t)


# In[ ]:


def get_accuracy(model,dataloader):
    count=0
    correct=0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            labels = batch[1]
            preds=network(images)
            batch_correct=preds.argmax(dim=1).eq(labels).sum().item()
            batch_count=len(batch[0])
            count+=batch_count
            correct+=batch_correct
    model.train()
    return correct/count


# In[ ]:


lr=0.001
batch_size=1000
shuffle=True
epochs=10

network = Network()
loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
optimizer = optim.Adam(network.parameters(), lr=lr)

# set the network to training mode
network.train()
for epoch in range(epochs):
    for batch in loader:
        images = batch[0]
        labels = batch[1]
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch {0}: train set accuracy {1}'.format(epoch+1,get_accuracy(network,loader)))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
print('Epoch {0}: test set accuracy {1}'.format(epoch+1,get_accuracy(network,test_loader)))

