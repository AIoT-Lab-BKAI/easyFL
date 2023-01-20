from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import os
import wandb
import numpy as np

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.client_ids = [i for i in range(len(self.clients))]
        self.latest_personal_test_acc = [0. for client_id in self.client_ids]
        
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        return self.latest_personal_test_acc, 0
    
    def iterate(self, t, pool):
        self.selected_clients = sorted(self.sample())
        models, personal_accs = self.communicate(self.selected_clients, pool, t)
                
        for client_id, personal_acc in zip(self.selected_clients, personal_accs):
            self.latest_personal_test_acc[client_id] = personal_acc
            
        if not self.selected_clients: return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.mu = option['mu']
    
    def train(self, model, device, round):
        # global parameters
        model = model.to(device)
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                original_loss = self.calculator.get_loss(model, batch_data, device)
                # proximal term
                loss_proximal = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm-ps,2))
                loss = original_loss + 0.5 * self.mu * loss_proximal                #
                loss.backward()
                optimizer.step()
        return

    def reply(self, svr_pkg, device, round):
        model = self.unpack(svr_pkg)
        self.train(model, device, round)
        acc, loss = self.test(model, device=device)
        cpkg = self.pack(model, acc)
        return cpkg
