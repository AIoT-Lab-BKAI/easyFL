import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
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