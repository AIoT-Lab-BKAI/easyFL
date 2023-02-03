import copy
from .mp_fedbase import MPBasicServer, MPBasicClient
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torch
import numpy as np
import os
from pathlib import Path
import wandb

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.client_ids = [i for i in range(len(self.clients))]
        self.latest_personal_test_acc = [0. for client_id in self.client_ids]
        
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        return self.latest_personal_test_acc, 0
        
    def iterate(self, t, pool):
        self.selected_clients = sorted(self.sample())
        _, personal_accs = self.communicate(self.selected_clients, pool, t)
                
        for client_id, personal_acc in zip(self.selected_clients, personal_accs):
            self.latest_personal_test_acc[client_id] = personal_acc
        return
        
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.model = None

    def reply(self, svr_pkg, device, round):
        model = self.unpack(svr_pkg)
        if self.model is None:
            self.model = copy.deepcopy(model)
            
        self.train(self.model, device, round)
        fin_acc, _ = self.test(self.model, device=device)
        cpkg = self.pack(0, fin_acc)
        return cpkg
    