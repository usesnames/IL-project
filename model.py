import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from copy import deepcopy

import resnet_cifar

# Hyper Parameters
num_epochs = 70
batch_size = 128
learning_rate = 2
weight_decay = 0.00001
gamma = 0.2

class incrementalNet(nn.Module):
  def __init__(self, n_classes, tot_classes, finetuning=True):
    # Network architecture
    super(incrementalNet, self).__init__()
    self.resnet = resnet_cifar.resnet32(num_classes=tot_classes)
    self.n_classes = n_classes
    self.tot_classes = tot_classes
    self.n_known = 0
    self.finetuning = finetuning

    # Learning method
    self.loss = nn.BCEWithLogitsLoss()

  def forward(self, x, feature_extractor=False):
      x = self.resnet(x, feature_extractor)
      return x

  # def increment_classes(self):
  #   in_features = self.resnet.fc.in_features
  #   weight = self.resnet.fc.weight.data
  #   bias = self.resnet.fc.bias.data
  #   self.resnet.fc = nn.Linear(in_features, self.n_known+self.n_classes, bias=True)
  #   self.resnet.fc.weight.data[:self.n_known] = weight    
  #   self.resnet.fc.bias.data[:self.n_known] = bias 

  def update_representation(self, dataset, order):
      
    self.cuda()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                            shuffle=True)

    # Store network with pre-update parameters
    if self.n_known > 0 and self.finetuning == False:
      old_net = deepcopy(self.resnet)
      old_net.eval()

      #self.increment_classes()
    
    # Run network training
    optimizer = optim.SGD(self.resnet.parameters(), momentum=0.9, lr=learning_rate,
                                weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[49,63], gamma=gamma)
   
    for epoch in tqdm(range(num_epochs)):
      for images, labels in dataloader:
        images = images.to(device = "cuda")
        labels = labels.to(device = "cuda")

        #labels = [order.index(el) for el in labels]
        #labels = torch.tensor(labels)

        self.resnet.train(True)
        optimizer.zero_grad()
        g = self.forward(images, feature_extractor=False)
        
        labels = F.one_hot(labels, num_classes = self.tot_classes) #(self.n_known+self.n_classes))
        labels = labels.float().cuda()
      
        if self.n_known > 0 and self.finetuning == False:
          old_outputs = torch.sigmoid(old_net.forward(images))
          labels[:,order[0:self.n_known]] = old_outputs[:,order[0:self.n_known]]
        
        loss = self.loss(g, labels)

        loss.backward()
        optimizer.step()

        
      #print ('Epoch [%d/%d], Loss: %.5f, LR: %.2f' 
       #           %(epoch+1, num_epochs, loss.item(), scheduler.get_last_lr()[0]))
      scheduler.step() 