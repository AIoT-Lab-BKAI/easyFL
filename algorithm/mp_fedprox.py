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
        self.mu = option['mu']
    
    def test(self, dataflag='valid', device='cpu', round=None):
        dataset = self.train_data if dataflag=='train' else self.valid_data
        self.model = self.model.to(device)
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
        return eval_metric, loss
    
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
        cpkg = self.pack(model, 0)
        self.model = model
        return cpkg

