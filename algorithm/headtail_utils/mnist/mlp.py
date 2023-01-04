
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule


class ClientModel(FModule):
    def __init__(self):
        super().__init__()
        self.head = ClientHead()
        self.tail = ClientTail()

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x
    
    def freeze_grad(self):
        self.head.freeze_grad()
        self.tail.freeze_grad() 
        
        
class ClientHead(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        
    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = F.relu(self.fc1(x))
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    
    
class ClientTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc2(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    
    
class ServerTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
