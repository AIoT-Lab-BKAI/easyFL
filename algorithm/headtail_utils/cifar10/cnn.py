from torch import nn
from utils.fmodule import FModule
import numpy as np
import torch.nn.functional as F

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
        
    def get_features(self, x):
        x = self.head(x)
        return self.tail.get_features(x)
        

class ClientHead(FModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    def freeze_grad(self):
        self.head.freeze_grad()
        self.tail.freeze_grad()
    
    
class ClientTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def get_features(self, x):
        features = [x]
        x = F.relu(self.fc1(x))
        features.append(x)
        x = self.fc2(x)
        features.append(x)
        return features
            
    
class ServerTail(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def get_features(self, x):
        features = [x]        
        x = F.relu(self.fc1(x))
        # features.append(x)
        x = F.relu(self.fc2(x))
        # features.append(x)
        x = F.relu(self.fc3(x))
        features.append(x)
        x = F.relu(self.fc4(x))
        # features.append(x)
        x = self.fc5(x)
        features.append(x)
        return features
    

class ServerModel(ClientModel):
    def __init__(self):
        super().__init__()
        self.tail = ServerTail()
        
    def get_features(self, x):
        x = self.head(x)
        return self.tail.get_features(x)