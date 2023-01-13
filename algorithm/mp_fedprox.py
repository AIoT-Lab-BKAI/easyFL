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
        self.paras_name = ['mu']
        self.max_acc = 0
        return
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, per_accs = self.communicate(self.selected_clients, pool, t)
        if not self.selected_clients: return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        
        self.max_acc = max(self.max_acc, np.mean(per_accs))

        # wandb record
        if self.wandb:
            wandb.log(
                {
                    "Mean Client Accuracy": np.mean(per_accs),
                    "Std Client Accuracy":  np.std(per_accs),
                    "Max Testing Accuracy": self.max_acc
                }
            )
            
        print(f"Max Testing Accuracy: {self.max_acc:>.3f}")
        print(f"Mean of Client Accuracy: {np.mean(per_accs):>.3f}")
        print(f"Std of Client Accuracy: {np.std(per_accs):>.3f}")
        return
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
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
        fin_acc, _ = self.test(model, device=device)
        cpkg = self.pack(model, fin_acc)
        return cpkg

