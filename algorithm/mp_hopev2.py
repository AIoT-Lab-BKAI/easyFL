# H.O.P.E: How Our Proposal Effective ?

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
        return
  
    def iterate(self, t, pool):
        self.selected_clients = self.sample()        
        self.data_vol_this_round = np.sum([self.client_vols[cid] for cid in self.selected_clients])
        
        models, train_losses = self.communicate(self.selected_clients, pool, t)
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol_this_round for cid in self.selected_clients]
        new_model = self.aggregate(models, p = impact_factors)
        
        self.model = self.model + len(self.selected_clients)/len(self.clients) * (new_model - self.model)
        return
    
    def pack(self, client_id):
        mu = len(self.selected_clients)/len(self.clients)
        
        return {
            "model" : copy.deepcopy(self.model),
            "next_impact": self.client_vols[client_id]/self.data_vol_this_round,
            "mu": mu,
        }


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.latest_round = 0
        self.contribution = None
        self.last_ipft = 0
    
    def unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['mu'], received_pkg['next_impact']
    
    def reply(self, svr_pkg, device, current_round):
        global_model, mu, next_impact = self.unpack(svr_pkg)
        model, train_loss = self.train(global_model, mu, next_impact, device, current_round)
        cpkg = self.pack(model, train_loss)
        return cpkg
    
    def train(self, global_model: FModule, mu, next_impact, device, current_round):
        global_model = global_model.to(device)
        
        if self.contribution is None:
            complement_model = global_model
            self.latest_round = current_round
            add_on = global_model.zeros_like()
        else:
            self.contribution = self.contribution.to(device)
            complement_model = global_model - (1 - mu)**(current_round - self.latest_round) * self.contribution
            add_on = self.contribution * 1.0/(mu * self.last_ipft)
        
        surogate_global_model = complement_model + mu * next_impact * add_on
        surogate_global_model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, surogate_global_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        mean_loss = []
        for iter in range(self.epochs):
            losses = []
            for batch_idx, batch_data in enumerate(data_loader):
                global_target_loss = self.calculator.get_loss(surogate_global_model, batch_data, device)                    
                losses.append(global_target_loss.detach().cpu())
                
                optimizer.zero_grad()
                global_target_loss.backward()
                optimizer.step()
            mean_loss.append(np.mean(losses))
        
        local_model = (surogate_global_model - complement_model) * 1.0/(mu * next_impact)
        local_model = local_model.to("cpu")
        
        # Update individual information
        self.contribution = local_model * mu * next_impact
        self.last_ipft = next_impact
        self.latest_round = current_round
        
        return local_model, np.mean(mean_loss)
    