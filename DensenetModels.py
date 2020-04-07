import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x

##### TRANSFER TO BINARY #####

class DenseNet121_Binary(nn.Module):
  def __init__(self, classCount=2, isTrained=True, transfer=True):
	
    super(DenseNet121_Binary, self).__init__()

    self.densenet121 = DenseNet121(classCount=14, isTrained=True)

    if transfer:
        modelCheckpoint = torch.load('./models/m-25012018-123527.pth.tar')
        new_state_dict = OrderedDict()

        ##### Convert Parrallel to Single GPU Loading Model #####
        for k, v in modelCheckpoint['state_dict'].items():
            if 'module.' in k:
                name = k[7:] # remove `module.`
                name = name.replace('norm.', 'norm')
                name = name.replace('conv.', 'conv')
                name = name.replace('normweight', 'norm.weight')
                name = name.replace('convweight', 'conv.weight')
                name = name.replace('normbias', 'norm.bias')
                name = name.replace('normrunning_mean', 'norm.running_mean')
                name = name.replace('normrunning_var', 'norm.running_var')
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        self.densenet121.load_state_dict(new_state_dict)

    kernelCount = self.densenet121.densenet121.classifier[0].in_features

    self.densenet121.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    for parameter in self.densenet121.parameters():
        parameter.requires_grad = False
    for parameter in self.densenet121.densenet121.classifier.parameters():
        parameter.requires_grad = True

    model_parameters = filter(lambda p: p.requires_grad, self.densenet121.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Total Trainable Params {params}')

  def forward(self, x):
    x = self.densenet121(x)
    return x


class ResNet50(nn.Module):
    def __init__ (self, classCount=2, isTrained=True, freeze=True):
        
        super(ResNet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)
        
        self.resnet50.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2))
        
        if freeze:
            for parameter in self.resnet50.parameters():
                parameter.requires_grad = False
            for parameter in self.resnet50.fc.parameters():
                parameter.requires_grad = True

        model_parameters = filter(lambda p: p.requires_grad, self.resnet50.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Total Trainable Params {params}')

    def forward (self, x):
        x = self.resnet50(x)
        return x

        