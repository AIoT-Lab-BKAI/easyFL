from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F

class Model(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=256, dim_out=10):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def decoder(self, x):
        x = self.fc3(x)
        return x

    def encoder(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)