import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import resnet_cifar

# Hyper Parameters
num_epochs = 70
batch_size = 128
learning_rate = 2 #will be changed to 0.2 if loss = 'ce+bce'
weight_decay = 0.00001

class iCaRLNet(nn.Module):
  def __init__(self, n_classes, tot_classes, transform, loss = "bce+bce", classifier_name = "standard", verbose=True):
    # Network architecture
    super(iCaRLNet, self).__init__()
    self.resnet = resnet_cifar.resnet32(num_classes=tot_classes)
    self.n_classes = n_classes
    self.tot_classes = tot_classes
    self.n_known = 0
    self.transform = transform
    self.order = []
    self.verbose = verbose
    self.classes_features = []
    self.classes_labels = []
    self.initialize_classifier = True
    self.classifier_name=classifier_name
    self.ablation=False

    if self.classifier_name != 'standard':
      self.ablation=True
      self.string='Computing features'
    else:
      self.string='Computing mean of classes'

    self.loss = loss
    if self.loss == "bce+l2":
      self.class_loss = nn.BCEWithLogitsLoss()
      self.dist_loss = nn.MSELoss()
    elif self.loss == "ce+bce":
      self.class_loss = nn.CrossEntropyLoss()
      self.dist_loss = nn.BCEWithLogitsLoss()
    elif self.loss == "l2+bce":
      self.class_loss = nn.MSELoss()
      self.dist_loss = nn.BCEWithLogitsLoss() 
    elif self.loss == 'bce+bce':
      self.class_loss = nn.BCEWithLogitsLoss()
    else:
      raise Exception("loss not found, possible values are 'bce+bce', 'ce+bce', 'l2+bce', 'bce+l2', you wrote {}".format(self.loss))
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
    if self.classifier_name == 'knn':
      preds = self.classify_KNN(x)
    elif self.classifier_name == 'svm':
      preds = self.classify_SVM(x)
    elif self.classifier_name == 'tree':
      preds = self.classify_tree(x)
    elif self.classifier_name == 'standard':
      preds = self.classify_standard(x)
    else:
      preds = None
      raise Exception("Classifier not found, possible values are 'standard', 'knn', 'svm', 'tree', you wrote {}".format(self.classifier_name))
    return preds

  
  def classify_standard(self, x):
    """Classify images by neares-means-of-exemplars"""
    batch_size = x.size(0)
    exemplar_means = self.exemplar_means
    means = torch.stack(exemplar_means) # (n_classes, feature_size)
    means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
    means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

    x = x.to(device="cuda")
    with torch.no_grad():
      feature = self.forward(x, feature_extractor=True) # (batch_size, feature_size)
    feature.data = feature.data/ feature.data.norm() 
    feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
    feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

    dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
    _, preds = dists.min(1)

    return torch.FloatTensor([self.order[i] for i in preds])


  def classify_tree(self, x):
    self.resnet.train(False)
    if self.initialize_classifier:
      #self.classifier = DecisionTreeClassifier(min_samples_split=(2000/self.n_known)) 
      self.classifier = RandomForestClassifier(min_samples_leaf=20)
      self.classifier.fit(self.classes_features, self.classes_labels)
      self.initialize_classifier = False    
    with torch.no_grad():
      feature = self.resnet(x,feature_extractor=True)  
    feature_to_predict=[]
    for i in range(feature.size(0)):
      feature.data[i] = feature.data[i]/ feature.data[i].norm()        
      lista=feature.data[i].tolist()
      feature_to_predict.append(lista)
    preds=self.classifier.predict(feature_to_predict)
    preds = torch.tensor(preds)
    return preds


  def classify_SVM(self, x):
    self.resnet.train(False)
    if self.initialize_classifier:  
      self.classifier = LinearSVC(random_state=1997, C=1.0) 
      self.classifier.fit(self.classes_features, self.classes_labels)
      self.initialize_classifier = False    
    with torch.no_grad():
      feature = self.resnet(x,feature_extractor=True)  
    
    feature_to_predict=[]
    for i in range(feature.size(0)):      
      feature.data[i] = feature.data[i]/ feature.data[i].norm()
      lista=feature.data[i].tolist()
      feature_to_predict.append(lista)
    preds=self.classifier.predict(feature_to_predict)
    preds = torch.tensor(preds)
    return preds


  def classify_KNN(self, x):
    self.resnet.train(False)    
    if self.initialize_classifier:
      self.classifier = KNeighborsClassifier(n_neighbors=10)
      self.classifier.fit(self.classes_features, self.classes_labels)        
      self.initialize_classifier = False       
    with torch.no_grad():  
      feature = self.resnet(x,feature_extractor=True)  #tensore[128,64]
    feature_to_predict=[]
    for i in range(feature.size(0)):
      feature.data[i] = feature.data[i]/ feature.data[i].norm()
      lista=feature.data[i].tolist()
      feature_to_predict.append(lista)
    preds=self.classifier.predict(feature_to_predict)
       
    return torch.tensor(preds)
      

  def compute_means_and_features(self, dataset):
    classes_features = []
    classes_labels = []
    exemplar_means = []
    self.resnet.eval()
    with torch.no_grad():
      for i in tqdm(range(0, self.n_known),
                    desc=self.string):
        features = []
        if i in range(0, (self.n_known-self.n_classes)) or self.ablation == True:
          for ex in self.exemplar_sets[i]:
            ex = self.transform(Image.fromarray(ex)).cuda()
            feature = self.resnet(ex.unsqueeze(0),feature_extractor=True)
            feature = feature.squeeze()
            feature.data = feature.data/ feature.data.norm()
            features.append(feature)
            classes_features.append(feature.tolist())
            classes_labels.append(self.order[i])
        else:
          images = dataset.get_image_class(self.order[i])
          for img in images:
            img = self.transform(Image.fromarray(img)).cuda()
            feature = self.resnet(img.unsqueeze(0),feature_extractor=True)
            feature = feature.squeeze()
            feature.data = feature.data/ feature.data.norm()
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
    """Construct an exemplar set for image set"""
    # Compute and cache features for each example
    self.resnet.train(False)
    features = []
    for img in images:
      img = self.transform(Image.fromarray(img))
      x = img.unsqueeze(0).to(device="cuda")
      feature = self.forward(x, feature_extractor=True).data.cpu().numpy()
      feature = feature / np.linalg.norm(feature) 
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


  def update_representation(self, dataset, order):
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
    if self.loss == 'ce+bce':
      learning_rate = 0.2
    else:
      learning_rate = 2 

    optimizer = optim.SGD(self.resnet.parameters(), momentum=0.9, lr=learning_rate,
                                weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[49,63], gamma=0.2)

    for epoch in tqdm(range(num_epochs), desc="Training"):
      for images, labels in dataloader:
        images = images.to(device = "cuda")
        labels = labels.to(device = "cuda")

        #labels = [order.index(el) for el in labels]
        #labels = torch.tensor(labels)

        self.resnet.train(True)
        optimizer.zero_grad()
        g = self.forward(images, feature_extractor=False)
        
        if self.loss == "bce+bce":
          labels = F.one_hot(labels, num_classes = self.tot_classes) #(self.n_known+self.n_classes))
          labels = labels.float().cuda()
        
          if self.n_known > 0:
            old_outputs = torch.sigmoid(old_net.forward(images))
            labels[:,order[0:self.n_known]] = old_outputs[:,order[0:self.n_known]]
        
          loss = self.class_loss(g, labels)

        elif self.loss == "ce+bce":
          d_loss=0.0
          
          if self.n_known > 0:
            old_outputs = torch.sigmoid(old_net.forward(images))
            d_loss = self.dist_loss(g[:,order[0:self.n_known]], old_outputs[:,order[0:self.n_known]])
         
          c_loss=self.class_loss(g, labels)
          loss = d_loss+c_loss

        elif self.loss == "l2+bce":
          labels = F.one_hot(labels, num_classes = self.tot_classes) 
          labels = labels.float().cuda()
          d_loss=0.0
          
          if self.n_known > 0:
            old_outputs = torch.sigmoid(old_net.forward(images))
            d_loss = self.dist_loss(g[:,order[0:self.n_known]], old_outputs[:,order[0:self.n_known]])
         
          softmax=torch.nn.Softmax(dim=1)
          g=softmax(g[:,order[self.n_known:(self.n_known+10)]])
          c_loss = self.class_loss(g, labels[:,order[self.n_known:(self.n_known+10)]])
          loss = d_loss+c_loss
        
        elif self.loss == "bce+l2":
          labels = F.one_hot(labels, num_classes = self.tot_classes) #(self.n_known+self.n_classes))
          labels = labels.float().cuda()
          d_loss=0.0
        
          if self.n_known > 0:
            old_outputs = torch.sigmoid(old_net.forward(images))
            d_loss = self.dist_loss(torch.sigmoid(g[:,order[0:self.n_known]]), old_outputs[:,order[0:self.n_known]])
        
          c_loss = self.class_loss(g[:,order[self.n_known:(self.n_known+10)]], labels[:,order[self.n_known:(self.n_known+10)]])
          loss = d_loss+c_loss

        loss.backward()
        optimizer.step()

      if self.verbose == True:  
        print ('Epoch [%d/%d], Loss: %.5f, LR: %.2f' 
                  %(epoch+1, num_epochs, loss.item(), scheduler.get_last_lr()[0]))
      scheduler.step() 