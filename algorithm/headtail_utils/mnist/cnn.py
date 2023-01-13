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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    
    
class ClientTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    
class ServerTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3136, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    