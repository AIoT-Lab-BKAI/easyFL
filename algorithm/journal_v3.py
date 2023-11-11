from .fedbase import BasicServer, BasicClient
from algorithm.agg_utils.fedtest_utils import get_penultimate_layer
from algorithm.drl_utils.ddpg import DDPG_Agent, KL_divergence

import torch.nn as nn
import numpy as np

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
        self.agent = DDPG_Agent(state_dim=(len(self.clients), 10, 256),
                                action_dim=self.clients_per_round)
        self.storage_path = option['storage_path']
        self.load_agent = option['load_agent']
        self.save_agent = option['save_agent']
        # self.is_infer= option['ep'] == 'infer'
        self.is_infer = False
        self.paras_name = ['ep']
        self.epsilon = 2
        self.prev_loss = 0
        return

    def run(self):
        if self.load_agent:
            self.agent.load_models(path=os.path.join(
                self.storage_path, "meta_models", "checkpoint_gamma0.001_rnn"))
            if self.is_infer:
                print("Freeze the StateProcessor!")
                self.agent.state_processor_frozen = True
            else:
                print("Unfreeze the StateProcessor!")
                self.agent.state_processor_frozen = False

        super().run()
        return

    def iterate(self, t):
        if t > 0 :
            _, losses_after_aggre, _ = self.communicate(self.selected_clients)
            print("Client loss after aggreation:", losses_after_aggre)
            # prev_sum_client = sum([self.client_vols[id] for id in self.selected_clients])
            curr_loss = np.mean(losses_after_aggre) + self.epsilon*(np.max(losses_after_aggre) - np.min(losses_after_aggre))
            prev_reward = self.prev_loss/curr_loss
        else:
            prev_reward = None

        self.selected_clients = sorted(self.sample())
        models, train_losses, _ = self.communicate(self.selected_clients)
        classifier_diffs = [get_penultimate_layer(
            self.model) * 0 for _ in self.clients]

        #Compute previous loss after aggregation

        for client_id, model in zip(self.selected_clients, models):
            classifier_diffs[client_id] = get_penultimate_layer(
                model.to(self.device) - self.model)

        raw_state = torch.stack(classifier_diffs, dim=0)
    
        impact_factor = self.agent.get_action(
            raw_state, self.selected_clients, prev_reward if t > 0 else None, log=self.wandb)
            
        if not self.selected_clients:
            return
    
        # ip = impact_factor[torch.tensor(self.selected_clients)]
        ip = impact_factor
        print("Clients: ", self.selected_clients)
        print("Impact factor:", ip.detach().cpu().numpy())
        self.model = self.aggregate(models, p=ip)
        self.prev_loss = (np.mean(train_losses) + self.epsilon*(np.max(train_losses) - np.min(train_losses)))
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