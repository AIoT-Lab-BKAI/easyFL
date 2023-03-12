import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.linear = nn.Linear(1000, 10)


    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

    

    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)