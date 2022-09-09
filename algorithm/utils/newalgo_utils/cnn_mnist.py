from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Feature_generator(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)

    def forward(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
def init():
    return (Feature_generator(), Classifier())
    
def forward(model, x):
    """
    model = (Feature_generator, Classifier)
    """
    x = model[0](x)
    x = model[1](x)
    return x

def pred_and_rep(model, x):
    """
    model = (Feature_generator, Classifier)
    """
    e = model[0](x)
    o = model[1](e)
    return o, e