from .mp_fedbase import MPBasicServer, MPBasicClient
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


def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    sim = []
    for layer_a, layer_b in zip(a.parameters(), b.parameters()):
        x, y = torch.flatten(layer_a), torch.flatten(layer_b)
        sim.append((x.T @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)

        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)

        self.impact_factor = None
        self.thr = 0.975
        self.optimal_ = np.array([1/6] * 6 + [1] * 4)
        

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        print("Selected:", self.selected_clients)
        models, train_losses = self.communicate(self.selected_clients, pool)
        models = [model.to(torch.device(f"cuda:{self.server_gpu_id}")) for model in models]
        if not self.selected_clients:
            return
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.update_Q_matrix(models, self.selected_clients, t)
            self.impact_factor = self.get_impact_factor(self.selected_clients, t)
        
        # with open("algorithm/invest/opt_loss.txt", "a+") as file:
        #     loss = np.mean(np.power(np.array(self.impact_factor) - self.optimal_[self.selected_clients], 2))
        #     file.write(f'{loss}\n')
            
        self.model = self.aggregate(models, p = self.impact_factor)
        self.update_threshold(t)
        return

    @torch.no_grad()
    def update_Q_matrix(self, model_list, client_idx, t=None):
        device = torch.device(f"cuda:{self.server_gpu_id}")
        self.model = self.model.to(device)
        models = []
        
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            models.append(model)
        
        new_similarity_matrix = torch.zeros_like(self.Q_matrix)
        for i, model_i in zip(client_idx, model_list):
            for j, model_j in zip(client_idx, model_list):
                _ , new_similarity_matrix[i][j] = compute_similarity(model_i, model_j)
                
        new_freq_matrix = torch.zeros_like(self.freq_matrix)
        for i in client_idx:
            for j in client_idx:
                new_freq_matrix[i][j] = 1
        
        # Increase frequency
        self.freq_matrix += new_freq_matrix
        self.Q_matrix = self.Q_matrix + new_similarity_matrix
        return

    @torch.no_grad()
    def get_impact_factor(self, client_idx, t=None):
        
        Q_asterisk_mtx = self.Q_matrix/(self.freq_matrix)
        Q_asterisk_mtx[torch.isinf(Q_asterisk_mtx)] = 0.0
        Q_asterisk_mtx = torch.nan_to_num(Q_asterisk_mtx, 0.0)
        
        min_Q = torch.min(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        max_Q = torch.max(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        Q_asterisk_mtx = torch.abs((Q_asterisk_mtx - min_Q)/(max_Q - min_Q) * (self.freq_matrix > 0.0))
        
        # if t % 2 == 0:
        #     np.savetxt(f"algorithm/invest/Q_matrix/Q_matrix_{t}.txt", Q_asterisk_mtx.numpy(), fmt='%.5f', delimiter=',')

        Q_asterisk_mtx = Q_asterisk_mtx > self.thr
        
        if t % 2 == 0:
            np.savetxt(f"algorithm/invest/Cluster/Cluster_mtx_{t}.txt", Q_asterisk_mtx.numpy(), fmt='%.5f', delimiter=',')
            
        # print(Q_asterisk_mtx * 1.0)
        impact_factor = 1/torch.sum(Q_asterisk_mtx, dim=0)
        impact_factor[torch.isinf(impact_factor)] = 0.0
        impact_factor = torch.nan_to_num(impact_factor, 0.0)
        impact_factor = impact_factor[client_idx].detach().cpu().tolist()
        # print(impact_factor)
        return impact_factor
    
    
    def update_threshold(self, t):
        self.thr = min(self.thr * (1 + 0.0005)**t, 0.998)
        return

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        
        
    def train(self, model, device):
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