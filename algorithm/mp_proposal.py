from .mp_fedbase import MPBasicServer, MPBasicClient
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torch
import numpy as np
import os
from pathlib import Path
import copy
from utils.fmodule import FModule


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.data_vol_this_round = None
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        self.data_vol_this_round = np.sum([self.client_vols[cid] for cid in self.selected_clients])
        
        models, train_losses = self.communicate(self.selected_clients, pool, t)
        if not self.selected_clients: return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol_this_round for cid in self.selected_clients]
        self.model = self.aggregate(models, p = impact_factors)
        return
    
    def pack(self, client_id):
        return {
            "model" : copy.deepcopy(self.model),
            "impact": 1.0 * self.client_vols[client_id]/self.data_vol_this_round,
        }

        
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.local_model = None
    
    def unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['impact']
    
    def reply(self, svr_pkg, device, round):
        global_model, pk = self.unpack(svr_pkg)
        loss = self.train_loss(global_model, device, round)
        model = self.train(global_model, pk, device, round)
        cpkg = self.pack(model, loss)
        return cpkg
    
    def train(self, global_model: FModule, pk, device, round):
        # Initialize model
        if self.local_model is None:
            self.local_model = copy.deepcopy(global_model).to(device)
        else:
            self.local_model = self.local_model.to(device)
        
        # Find complement model
        global_model = global_model.to(device)
        with torch.no_grad():
            complement_model = global_model - pk * self.local_model
        
        surogate_global_model = complement_model + pk * self.local_model
        surogate_global_model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, surogate_global_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                global_target_loss = self.calculator.get_loss(surogate_global_model, batch_data, device)                    
                
                optimizer.zero_grad()
                global_target_loss.backward()
                optimizer.step()
                        
        return (surogate_global_model - complement_model) * 1/pk
    