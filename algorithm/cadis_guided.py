"""
This version is an extension from fedsdivv2
The update model at server is modified as: 
    new_model = aggregate(delta_models, impact_factors)
    server_model = server_model + lr * z * new_model

In which: lr = client_per_turn / all_clients
          z  = new_clients_this_turn / client_per_turn
"""
from .fedbase import BasicServer, BasicClient
from algorithm.agg_utils.fedtest_utils import get_module_from_model
from algorithm.cfmtx.cfmtx import cfmtx_test
from algorithm.drl_utils.networks import StateProcessor
from algorithm.drl_utils.buffer import ReplayBuffer, Memory
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn as nn
import numpy as np

import torch
import os
import copy
# import wandb, time, json


class StateDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        return
    
    def insert(self, x):
        self.data.append(x)     # N x M x d
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]   # N x M x d
    

def get_penultimate_layer(model):
    penul = get_module_from_model(model)[-1]._parameters['bias']
    return penul


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
    sub_s_norm = sub_s_norm.flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


@torch.no_grad()
def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    pen_a = torch.flatten(get_penultimate_layer(a))
    pen_b = torch.flatten(get_penultimate_layer(b))
    return (pen_a @ pen_b) / (torch.norm(pen_a) * torch.norm(pen_b))

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)

        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)
        
        self.impact_factor = None
        self.thr = option['sthr']
        
        self.gamma = 1
        self.device = torch.device("cuda")
        
        self.paras_name = ['kd_fct', 'sthr']
        self.once_time_clients = []
        
        self.memory = Memory()
        self.replay_buffer = ReplayBuffer(10000)
        self.state_dataset = StateDataset()
        self.state_processor = StateProcessor(state_dim=(len(self.clients), 10, 256), hidden_dim=128).cuda()
        return
        
    def run(self):
        torch.save(self.state_processor.state_dict(), os.path.join("./storage/models", "StateProcessor.pt"))
        super().run()
        self.replay_buffer.save(path=os.path.join("./storage", "buffers"), name=self.task + "_CADIS-GUIDED")
        return
    
    def sp_update(self, device='cuda', update=True):
        optimizer = optim.SGD(self.state_processor.parameters(), lr=1e-3)
        loader = DataLoader(self.state_dataset, batch_size=4, drop_last=False, shuffle=True)
        losses = []
        for epoch in range(8):
            epoch_losses = []
            for batch_state in loader:
                batch_state = batch_state.clone().detach().to(device)                           # batch x N x M x d
                batch_processed_state = self.state_processor(batch_state).transpose(1,2)        # batch x N x M
                correlation_loss = 0
                for state, pstate in zip(batch_state, batch_processed_state):
                    correlation_loss += KL_divergence(state.flatten(1,2), pstate, device)
                
                optimizer.zero_grad()
                correlation_loss.backward()                                                     # type:ignore
                if update:
                    optimizer.step()
                    
                epoch_losses.append(correlation_loss.detach().cpu().item())                     # type:ignore
            losses.append(np.mean(epoch_losses))
        return np.mean(losses)
    
    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        
        classifier_diffs = [get_penultimate_layer(self.model) * 0 for _ in self.clients]
        for client_id, model in zip(self.selected_clients, models):
            classifier_diffs[client_id] = get_penultimate_layer(model - self.model)
        
        raw_state = torch.stack(classifier_diffs, dim=0)
        self.state_dataset.insert(raw_state)                     # put into dataset: N x M x d
        preprocessed_state = self.state_processor(raw_state.cuda())
        if t  > 0:
            prev_reward = - np.mean(train_losses)
            self.memory.update(r=prev_reward)
            if len(self.replay_buffer) >= 8:
                self.sp_update()
        
        models = [model.to(self.device) for model in models]
        self.model = self.model.to(self.device)                                     # type: ignore
        model_diffs = [model.to(self.device) - self.model for model in models]
        
        self.update_Q_matrix(model_diffs, self.selected_clients, t)
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.impact_factor, self.gamma = self.get_impact_factor(self.selected_clients, t)
        self.model = self.aggregate(models, p=self.impact_factor)
        
        self.update_threshold(t)
        for cid in self.selected_clients: 
            if cid not in self.once_time_clients:
                self.once_time_clients.append(cid)
                
        self.memory.act(preprocessed_state.detach().cpu(), self.impact_factor)
        if self.memory.get_last_record():
            s, a, r, s_next = self.memory.get_last_record()    # type: ignore
            self.replay_buffer.push(s, a, r, s_next, 0)
        return
    
    def compute_simXY(self, simXZ, simZY):
        sigma = torch.abs(torch.sqrt((1-simXZ**2) * (1-simZY**2)))
        return simXZ * simZY, sigma 

    def transitive_update_Q(self):
        temp_Q = torch.zeros_like(self.Q_matrix)
        temp_F = torch.zeros_like(self.freq_matrix)
        
        for i in range(len(self.clients)):
            for j in range(i+1, len(self.clients)):
                for k in range(j+1, len(self.clients)):
                    if (self.Q_matrix[i,j] != 0) and (self.Q_matrix[i,k] != 0) and (self.Q_matrix[j,k] == 0):
                        simi, sigma = self.compute_simXY(self.Q_matrix[i,j]/self.freq_matrix[i,j],
                                                        self.Q_matrix[i,k]/self.freq_matrix[i,k])
                        if sigma < 0.02 and simi > self.thr:
                            temp_Q[j,k] += simi
                            temp_F[j,k] += 1
                            temp_Q[k,j] = temp_Q[j,k]
                            temp_F[k,j] += 1
                            # print(f"Transitive: Client[{j}] - Client[{k}], By Client[{i}]: {simi:>.5f}")
                    
                    elif (self.Q_matrix[i,j] != 0) and (self.Q_matrix[i,k] == 0) and (self.Q_matrix[j,k] != 0):
                        simi, sigma = self.compute_simXY(self.Q_matrix[i,j]/self.freq_matrix[i,j],
                                                        self.Q_matrix[j,k]/self.freq_matrix[j,k])
                        if sigma < 0.02 and simi > self.thr:
                            temp_Q[i,k] += simi
                            temp_F[i,k] += 1
                            temp_Q[k,i] = temp_Q[i,k]
                            temp_F[k,i] += 1
                            # print(f"Transitive: Client[{i}] - Client[{k}], By Client[{j}]: {simi:>.5f}")
                    
                    elif (self.Q_matrix[i,j] == 0) and (self.Q_matrix[i,k] != 0) and (self.Q_matrix[j,k] != 0):
                        simi, sigma = self.compute_simXY(self.Q_matrix[i,k]/self.freq_matrix[i,k],
                                                        self.Q_matrix[j,k]/self.freq_matrix[j,k])
                        if sigma < 0.02 and simi > self.thr:
                            temp_Q[i,j] += simi
                            temp_F[i,j] += 1
                            temp_Q[j,i] = temp_Q[i,j]
                            temp_F[j,i] += 1
                            # print(f"Transitive: Client[{j}] - Client[{i}], By Client[{k}]: {simi:>.5f}")
                        
        temp_Q[temp_Q > 0] = temp_Q[temp_Q > 0]/temp_F[temp_Q > 0]
        self.Q_matrix += temp_Q
        self.freq_matrix += (temp_F > 0) * 1.0
        return

    def remove_inf_nan(self, input):
        input[torch.isinf(input)] = 0.0
        input = torch.nan_to_num(input, 0.0)
        return input

    @torch.no_grad()
    def update_Q_matrix(self, model_list, client_idx, t=None):
        new_similarity_matrix = torch.zeros_like(self.Q_matrix)
                
        for i, model_i in zip(range(len(client_idx)), model_list):
            new_similarity_matrix[client_idx[i]][client_idx[i]] = 1
            for j, model_j in zip(range(i+1, len(client_idx)), model_list):
                new_similarity_matrix[client_idx[i]][client_idx[j]] = compute_similarity(model_i, model_j)
                new_similarity_matrix[client_idx[j]][client_idx[i]] = new_similarity_matrix[client_idx[i]][client_idx[j]]

        new_freq_matrix = torch.zeros_like(self.freq_matrix)

        for i in range(len(client_idx)):
            new_freq_matrix[client_idx[i]][client_idx[i]] = 1
            for j in range(i+1, len(client_idx)):
                new_freq_matrix[client_idx[i]][client_idx[j]] = 1
                new_freq_matrix[client_idx[j]][client_idx[i]] = 1

        # Increase frequency
        self.freq_matrix += new_freq_matrix
        self.Q_matrix = self.Q_matrix + new_similarity_matrix
        
        # if len(self.selected_clients) < len(self.clients):
        #     self.transitive_update_Q()
        return

    @torch.no_grad()
    def get_impact_factor(self, client_idx, t=None):      
        
        Q_asterisk_mtx = self.Q_matrix/(self.freq_matrix)
        Q_asterisk_mtx = self.remove_inf_nan(Q_asterisk_mtx)
        
        min_Q = torch.min(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        max_Q = torch.max(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        Q_asterisk_mtx = torch.abs((Q_asterisk_mtx - min_Q)/(max_Q - min_Q) * (self.freq_matrix > 0.0))
        
        Q_asterisk_mtx = (Q_asterisk_mtx > self.thr) * 1.0
        Q_asterisk_mtx = ((Q_asterisk_mtx.T @ Q_asterisk_mtx) > 0) * 1.0        # Enhance Transitive clustering matrix
        
        impact_factor = 1/torch.sum(Q_asterisk_mtx, dim=0)
        impact_factor = self.remove_inf_nan(impact_factor)
        impact_factor_frac = impact_factor[client_idx]
        
        num_cluster_all = torch.sum(impact_factor)
        
        temp_mtx = Q_asterisk_mtx[client_idx]
        temp_mtx = temp_mtx.T
        temp_mtx = temp_mtx[client_idx]
        
        temp_vec = 1/torch.sum(temp_mtx, dim=0)
        temp_vec = self.remove_inf_nan(temp_vec)
        
        num_cluster_round = torch.sum(temp_vec)
        gamma = num_cluster_round/num_cluster_all
        
        return impact_factor_frac.detach().cpu().tolist(), gamma.detach().cpu().item()
    
    def update_threshold(self, t):
        self.thr = min(self.thr * (1 + 0.0005)**t, 0.985)
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
    
    def data_to_device(self, data, device):
        return data[0].to(device), data[1].to(device)

    def get_loss(self, model, src_model, data, device):
        tdata = self.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KL_divergence(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss