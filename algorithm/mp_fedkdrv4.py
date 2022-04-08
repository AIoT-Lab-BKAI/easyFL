""" FedKDR - Version 4
This version combines Reinforcement learning with FedKDR
Details:
    The impact factor of the client is calculated as:
        p_i = 1 + sqrt(N_i / (ln(t) + 1)) -> Confidence
    
    The new model is aggregated as:
        new_model = sum(p_i * model_i)
    
    The new model is cumulatively updated into the server model by factor beta
        model <- (1 - beta) * model + beta * new_model
    
    The purpose of RL agent is to compute beta
"""

from pathlib import Path
from utils import fmodule
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.fedrl_utils.gae_agent.gae_agent import gae_agent

import torch.nn as nn
import numpy as np

import torch
import os
import copy


def KL_divergence(teacher_batch_input, student_batch_input, device):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    Author: Nguyen Nang Hung
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.to(device).unsqueeze(1)
    student_batch_input = student_batch_input.to(device).unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm[sub_s_norm!=0].view(batch_student,-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm[sub_t_norm!=0].view(batch_teacher,-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.agent = gae_agent(
            num_inputs=2*len(self.clients), 
            num_outputs=1, 
            hidden_size=256, 
            device=torch.device(f'cuda:{self.server_gpu_id}')
            )
        
        self.frequency_record = torch.zeros((1, len(self.clients)))
        self.prev_reward = 0
        return
    
    def finish(self, model_path):
        if not Path(model_path).exists():
            os.system(f"mkdir -p {model_path}")
        task = self.option['task']
        torch.save(self.model.state_dict(), f"{model_path}/{self.name}_{self.num_rounds}_{task}.pth")
        return
    
    def run(self):
        super().run()
        # self.finish(f"algorithm/fedrl_utils/baseline/{self.name}")
        return
    
    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        frequencies = [cp["frequency"] for cp in packages_received_from_clients]
        return models, train_losses, frequencies
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        self.selected_clients = [0,1]
        models, train_losses, frequencies = self.communicate(self.selected_clients, pool)
        
        self.prev_reward = - np.sum(train_losses) if t > 0 else None
        
        if not self.selected_clients: 
            return
        else:
            device0 = torch.device(f"cuda:{self.server_gpu_id}")
            models = [i.to(device0) for i in models]
            
            idx_one_hot = self.onehot_fromlist(self.selected_clients)
            self.frequency_record += idx_one_hot
            
            state = torch.vstack([idx_one_hot, self.frequency_record]).flatten().to(device0)
            incremental_factor = self.agent.get_action(state, self.prev_reward).cpu()
            
            new_model = self.aggregate(models, p = [(1 + np.sqrt(f/(np.log(t+1) + 1))) for f,cid in zip(frequencies, self.selected_clients)])
            self.model = fmodule._model_add(self.model.cpu() * incremental_factor.detach(), new_model.cpu() * (1 - incremental_factor.detach()))
            return

    def onehot_fromlist(self, list, length=None):
        if length == None:
            length = len(self.clients)
        output = torch.zeros([1, length])
        for idx in list:
            output[0, idx] = 1
        return output
    

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.frequency = 0

        
    def train(self, model, device):
        self.frequency += 1
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
                
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(model, src_model, batch_data, device)
                loss = loss + kl_loss
                loss.backward()
                optimizer.step()
        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, src_model, data, device):
        tdata = self.data_to_device(data, device)
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KL_divergence(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss
    
    
    def pack(self, model, loss, frequency):
        return {
            "model": model,
            "train_loss": loss,
            "frequency": frequency
        }
    
    
    def reply(self, svr_pkg, device):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)
        cpkg = self.pack(model, loss, self.frequency)
        return cpkg