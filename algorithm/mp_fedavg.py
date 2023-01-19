from .mp_fedbase import MPBasicServer, MPBasicClient
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torch
import numpy as np
import os
from pathlib import Path
import copy

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(dataflag, device=device, round=round)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses
        
class Client(MPBasicClient):
    def __init__(self, option, name='', init_model=None, train_data=None, valid_data=None):
        super().__init__(option, name, init_model, train_data, valid_data)
        self.base_model = copy.deepcopy(self.model)
    
    def test(self, dataflag='valid', device='cpu', round=None):
        dataset = self.train_data if dataflag=='train' else self.valid_data
        
        self.model = self.model.to(device)
        diff = self.model - self.base_model.to(device)
        
        print("Client", self.name, f"diff: {diff.norm():>.5f}", end=" -> ")
        
        self.model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(self.model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        
        print("eval:", eval_metric)
        return eval_metric, loss
    
    def reply(self, svr_pkg, device, round):
        self.model = self.unpack(svr_pkg)
        self.train(self.model, device, round)
        diff = self.model - self.base_model.to(device)
        print("Client", self.name, f"done training, diff: {diff.norm():>.3f}")
        cpkg = self.pack(self.model, 0)
        return cpkg