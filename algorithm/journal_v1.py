from .fedbase import BasicServer, BasicClient
from algorithm.agg_utils.fedtest_utils import get_penultimate_layer
from algorithm.cfmtx.cfmtx import cfmtx_test
from algorithm.drl_utils.ddpg import DDPG_Agent, KL_divergence

import torch.nn as nn
import numpy as np

import torch
import os
import copy
import wandb, time, json


time_records = {"server_aggregation": {"overhead": [], "aggregation": []}, "local_training": {}}


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.device='cuda'
        self.agent = DDPG_Agent(state_dim=(len(self.clients), 10, 256),
                                action_dim=self.clients_per_round)
        self.storage_path=option['storage_path']
        self.load_agent=option['load_agent']
        self.save_agent=option['save_agent']
        return
    
    def run(self):
        if self.load_agent:
            self.agent.load_models(path=os.path.join(self.storage_path, "models"))
        self.agent.load_buffer(path=os.path.join(self.storage_path, "buffers"), discard_name=self.task)
        
        super().run()
        
        if self.save_agent:
            self.agent.save_models(path=os.path.join(self.storage_path, "models"))
        self.agent.save_buffer(path=os.path.join(self.storage_path, "buffers"), name=self.task)
        return
    
    def iterate(self, t):
        self.selected_clients = sorted(self.sample())
        models, train_losses = self.communicate(self.selected_clients)
        classifier_diffs = [get_penultimate_layer(self.model) * 0 for _ in self.clients]
        for client_id, model in zip(self.selected_clients, models):
            classifier_diffs[client_id] = get_penultimate_layer(model - self.model)
        
        raw_state = torch.concat(classifier_diffs, dim=0)
        print(raw_state.shape) # N x M x d
        prev_reward = - np.mean(train_losses)
        impact_factor = self.agent.get_action(raw_state, prev_reward if t > 0 else None)

        if not self.selected_clients:
            return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = impact_factor)
        return


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_fct = option['kd_fct']
        
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg
        
    def train(self, model, device='cuda'):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
                
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(model, src_model, batch_data, device)
                loss = loss + self.kd_fct * kl_loss
                loss.backward()
                optimizer.step()
        return

    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KL_divergence(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss