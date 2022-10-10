from cmath import isnan
import copy
import torch.nn.functional as F
from torch import nn
import torch
from utils.fmodule import FModule

@torch.no_grad()
def batch_similarity(a, b):
    """
    Args:
        a of shape (x, y)
        b of shape (z, y)
    return:
        c = sim (a, b) of shape (x, z)
    """
    a = a.cpu()
    up = (a @ b.T)
    down = (torch.norm(a, dim=1, keepdim=True) @ torch.norm(b, dim=1, keepdim=True).T)
    val = up / down
    val = torch.nan_to_num(val, 0)
    return val

class MnistCnn(FModule):
    def __init__(self, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10, bias=bias)
        self.lowrank_mtx = None
        self.Phi = None
        self.Psi = None
        return
    
    def forward(self, x):
        r_x = self.encoder(x)
        return self.masking(r_x)
    
    def pred_and_rep(self, x):
        r_x = self.encoder(x)
        output = self.masking(r_x)
        return output, r_x
    
    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x
    
    def masking(self, r_x):
        logits = self.fc2(r_x).unsqueeze(2)
        mask = None
        if self.lowrank_mtx is not None:
            mask = (self.lowrank_mtx @ self.lowrank_mtx).unsqueeze(0)
        else:
            b = r_x.shape[0]
            psi_x = batch_similarity(r_x, self.Psi) 
            mask = (self.Phi.view(self.Phi.shape[0], -1).T @ psi_x.unsqueeze(2)).view(b, 10, 10)
        return (mask.to("cuda" if logits.is_cuda else "cpu").to(torch.float32) @ logits).squeeze(2)

    def update_mask(self, mask):
        """
        Note: For client only
        """
        self.lowrank_mtx = mask
        return
    
    def prepare(self, Phi, Psi):
        self.Psi = Psi
        self.Phi = Phi
        self.lowrank_mtx = None
        return
    
    