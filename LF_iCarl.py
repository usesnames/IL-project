import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import modified_resnet_cifar

# Hyper Parameters
num_epochs = 70
batch_size = 128
learning_rate = 0.2
weight_decay = 0.00001

class LFiCaRLNet(nn.Module):
  def __init__(self, n_classes, tot_classes, transform, verbose=True):
    # Network architecture
    super(LFiCaRLNet, self).__init__()
    self.resnet = modified_resnet_cifar.resnet32(num_classes=tot_classes)
    self.n_classes = n_classes
    self.tot_classes = tot_classes
    self.n_known = 0
    self.transform = transform
    self.order = []
    self.verbose = verbose
    self.classes_features = []
    self.classes_labels = []
    self.initialize_classifier = True
    self.svm=False
    
    self.class_loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.CosineEmbeddingLoss()
    # List containing exemplar_sets
    # Each exemplar_set is a np.array of N images
    # with shape (N, C, H, W)
    self.exemplar_sets = []
    
    # Means of exemplars
    self.exemplar_means = []


  def forward(self, x, feature_extractor=False):
      x = self.resnet(x, feature_extractor)
      return x


  def classify(self, x):
    """Classify images by neares-means-of-exemplars

    Args:
        x: input image batch
    Returns:
        preds: Tensor of size (batch_size,)
    """
    batch_size = x.size(0)
    exemplar_means = self.exemplar_means
    means = torch.stack(exemplar_means) # (n_classes, feature_size)
    means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
    means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

    x = x.to(device="cuda")
    with torch.no_grad():
      feature = self.forward(x, feature_extractor=True) # (batch_size, feature_size)
    feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
    feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

    dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
    _, preds = dists.min(1)

    return torch.FloatTensor([self.order[i] for i in preds])
      

  def compute_means_and_features(self, dataset):
    classes_features = []
    classes_labels = []
    exemplar_means = []
    self.resnet.eval()
    with torch.no_grad():
      for i in tqdm(range(0, (self.n_known)),
                    desc="Computing mean of classes"):
        features = []
        if i in range(0, (self.n_known-self.n_classes)) or self.svm == True:
          for ex in self.exemplar_sets[i]:
            ex = self.transform(Image.fromarray(ex)).cuda()
            feature = self.resnet(ex.unsqueeze(0),feature_extractor=True)
            feature = feature.squeeze()
            
            features.append(feature)
            classes_features.append(feature.tolist())
            classes_labels.append(self.order[i])
        else:
          images = dataset.get_image_class(self.order[i])
          for img in images:
            img = self.transform(Image.fromarray(img)).cuda()
            feature = self.resnet(img.unsqueeze(0),feature_extractor=True)
            feature = feature.squeeze()
            
            features.append(feature)
            classes_features.append(feature.tolist())
            classes_labels.append(self.order[i])
        features = torch.stack(features)
        mu_y = features.mean(0).squeeze()
        mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
        exemplar_means.append(mu_y)
      self.exemplar_means = exemplar_means
      self.classes_features = classes_features
      self.classes_labels = classes_labels


  def construct_exemplar_set(self, images, m):
    """Construct an exemplar set for image set

    Args:
        images: np.array containing images of a class
    """
    # Compute and cache features for each example
    self.resnet.train(False)
    features = []
    for img in images:
      img = self.transform(Image.fromarray(img))
      x = img.unsqueeze(0).to(device="cuda")
      with torch.no_grad():
        feature = self.forward(x, feature_extractor=True).data.cpu().numpy()
      
      features.append(feature[0])

    features = np.array(features)
    class_mean = np.mean(features, axis=0)
    class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

    exemplar_set = []
    exemplar_features = [] # list of Variables of shape (feature_size,)
    for k in range(int(m)):
      S = np.sum(exemplar_features, axis=0)
      phi = features
      mu = class_mean
      mu_p = 1.0/(k+1) * (phi + S)
      mu_p = mu_p / np.linalg.norm(mu_p)
      i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

      exemplar_set.append(images[i])
      exemplar_features.append(features[i])
      # avoid duplicates in exemplars set
      images = np.delete(images, i, axis = 0)
      features = np.delete(features, i, axis = 0)
    
    self.exemplar_sets.append(np.array(exemplar_set)) 
            

  def reduce_exemplar_sets(self, m):
    for y, P_y in enumerate(self.exemplar_sets):
      self.exemplar_sets[y] = P_y[:int(m)]


  def combine_dataset_with_exemplars(self, dataset):
    for y, P_y in enumerate(self.exemplar_sets):
      exemplar_images = P_y
      exemplar_labels = [self.order[y] for i in range(len(P_y))]
      dataset.append(exemplar_images, exemplar_labels)


  def update_representation(self, dataset, order, device='cuda'):
    self.initialize_classifier = True
    self.order = order  
    
    self.cuda() 
    # Form combined training set
    self.combine_dataset_with_exemplars(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                            shuffle=True)
    # Store network with pre-update parameters
    if self.n_known > 0:
      old_net = deepcopy(self.resnet)
      old_net.eval()

      #self.increment_classes()
    
    # Run network training
    optimizer = optim.SGD(self.resnet.parameters(), momentum=0.9, lr=learning_rate,
                                weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[49,63], gamma=0.2)

    for epoch in tqdm(range(num_epochs), desc="Training"):
      cum_loss = 0.0
      steps = 0
      for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        self.resnet.train(True)
        optimizer.zero_grad()
        g = self.forward(images, feature_extractor=False)

        if self.n_known > 0:
          self.resnet.train(False)
          l = 6*(np.sqrt(self.n_known/self.n_classes))
          new_features = self.forward(images, feature_extractor=True)
          new_features = new_features.data/new_features.data.norm()
          old_features = old_net(images, feature_extractor=True)
          old_features = old_features.data/old_features.data.norm()

          loss1 = self.dist_loss(new_features, old_features, \
              torch.ones(images.shape[0]).to(device)) * l
          loss2 = self.class_loss(g, labels)
          loss = loss1 + loss2
        else:      
          loss = self.class_loss(g, labels)
        cum_loss += loss.item()
        steps += 1
        loss.backward()
        optimizer.step()

      if self.verbose == True:  
        print ('Epoch [%d/%d], Loss: %.5f, LR: %.2f' 
                  %(epoch+1, num_epochs, cum_loss/steps, scheduler.get_last_lr()[0]))
      scheduler.step() 