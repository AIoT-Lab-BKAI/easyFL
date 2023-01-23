from torch import nn
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
        self.decoder = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.decoder(x)
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    
class ServerTail(FModule):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        return self.decoder(x)
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            