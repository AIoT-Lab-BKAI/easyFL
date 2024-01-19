from .fedbase import BasicServer, BasicClient
from algorithm.agg_utils.fedtest_utils import get_penultimate_layer
from algorithm.utils_new.ddpg import DDPG_Agent, KL_divergence

import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import os
import copy
import math


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


time_records = {"server_aggregation": {
    "overhead": [], "aggregation": []}, "local_training": {}}


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.device = 'cuda'
        self.agent = DDPG_Agent(state_dim=(self.clients_per_round, 100, 256),
                                action_dim=self.clients_per_round)
        self.epsilon = option['eps']
        self.paras_name = ['eps', 'kd_fct']
        self.client_epochs = option['num_epochs']
        self.client_batch_size = option['batch_size']

        return

    # def run(self):
    #     if self.load_agent:
    #         self.agent.load_models(path=os.path.join(
    #             self.storage_path, "meta_models", "checkpoint_gamma0.001_rnn"))
    #         if self.is_infer:
    #             print("Freeze the StateProcessor!")
    #             self.agent.state_processor_frozen = True
    #         else:
    #             print("Unfreeze the StateProcessor!")
    #             self.agent.state_processor_frozen = False

    #     super().run()
    #     return

    def iterate(self, t):

        self.selected_clients = sorted(self.sample())
        models, train_losses, _ = self.communicate(self.selected_clients)
        # print("Client loss after aggreation:", train_losses)
        
        # Calculate reward
        client_vol_t = [self.client_vols[id] for id in self.selected_clients]
        sum_client =sum(client_vol_t)
        client_vol_t = [self.client_vols[id]/sum_client for id in self.selected_clients]
        mean_loss = sum([self.client_vols[id] * loss for id, loss in zip(self.selected_clients, train_losses)])/sum_client
        avg_loss = (np.mean(train_losses) + self.epsilon*(np.max(train_losses) - np.min(train_losses)))
        reward = np.exp(-avg_loss)

        gradients = []
        alphas = []
        classifier_diffs = []

        for client_id, model, loss in zip(self.selected_clients, models, train_losses):
            beta = self.client_epochs * math.ceil(self.client_vols[client_id] / self.client_batch_size)
            alphas.append(beta*self.client_vols[client_id]/sum_client)
            gradients.append((model.to(self.device) - self.model))
            
            grad = get_penultimate_layer(model.to(self.device) - self.model)
            classifier_diffs.append(grad) 

        raw_state = torch.stack(classifier_diffs, dim=0)

        # Tính min và max của list
        min_loss, max_loss = min(train_losses), max(train_losses)
        min_num_client, max_num_client = min(client_vol_t), max(client_vol_t)
        min_grad, max_grad = torch.min(raw_state), torch.max(raw_state)

        # Áp dụng Min-Max Scaling
        # scaled_losses = [(x - min_loss) / (max_loss - min_loss) for x in train_losses]
        # scaled_num_client = [(x - min_num_client) / (max_num_client - min_num_client) for x in client_vol_t]
        scaled_grad = (raw_state - min_grad) / (max_grad - min_grad)

        impact_factor = self.agent.get_action(
            (scaled_grad, train_losses, client_vol_t), reward if t > 0 else None, log=self.wandb)
            
        if not self.selected_clients:
            return

        # print("Clients: ", self.selected_clients)
        print("Impact factor:", impact_factor.detach().cpu().numpy())
        
        ip2 = impact_factor.detach().cpu().numpy()
        ip_final = [ip2[id] for id in range(len(ip2))]

        self.model = self.aggregate(models, p=ip_final)
        return
    
    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        after_train_losses = [cp["after_train_loss"] for cp in packages_received_from_clients] 
        return models, train_losses, after_train_losses   


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_fct = option['kd_fct']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)

        self.train(model)
        after_train_loss = self.train_loss(model)
        cpkg = self.pack(model, loss, after_train_loss)
        return cpkg
    
    def pack(self, model, loss, after_train_loss):
        return {
            "model" : model,
            "train_loss": loss,
            "after_train_loss": after_train_loss
        }

    def train(self, model, device='cuda'):
        model = model.to(device)
        model.train()

        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()

        data_loader = self.calculator.get_data_loader(
            self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(
            self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(
                    model, src_model, batch_data, device)
                loss = loss + self.kd_fct * kl_loss
                loss.backward()
                optimizer.step()
                
                src_model = copy.deepcopy(model).to(device)
                src_model.freeze_grad()
        return

    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)
        
        output_s, representation_s = model.pred_and_rep(
            tdata[0])                  # Student
        _, representation_t = src_model.pred_and_rep(
            tdata[0])                    # Teacher
        kl_loss = KL_divergence(
            representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        # print(loss, kl_loss)
        return loss, kl_loss