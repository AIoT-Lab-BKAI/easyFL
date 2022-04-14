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
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("model contains nan")
            exit(0)

        norm_x, norm_y = torch.norm(x), torch.norm(y)
        if norm_x == 0 or norm_y == 0:
            sim.append(0.0)
        else:
            m = (x.T @ y) / (norm_x * norm_y)
            sim.append(m)

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.thr = 0.99
        
        self.all_client_models = [copy.deepcopy(self.model).zeros_like() for _ in self.clients]
        self.last_all_impact_factor = [0 for _ in self.clients]
        self.optimal_ = np.array([1/6] * 6 + [1] * 4)

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        print(self.selected_clients)
        
        models, train_losses = self.communicate(self.selected_clients, pool)
        models = [model.to(torch.device(f"cuda:{self.server_gpu_id}")) for model in models]
        
        if not self.selected_clients:
            return
        
        impact_factor = self.update_global_model(models, self.selected_clients, t)
            
        # update model list
        self.update_all_client_models_list(models, self.selected_clients)
        self.last_all_impact_factor = list.copy(impact_factor)
        return


    @torch.no_grad()
    def update_global_model(self, model_list, idx_list, t):
        device = torch.device(f"cuda:{self.server_gpu_id}")
        self.model = self.model.to(device)
        
        new_models = []
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            new_models.append(model)
        
        old_models = []
        for model in self.all_client_models:
            for p, q in zip(model.parameters(), self.model.parameters()):
                 p = p - q
            old_models.append(model)
        
        similarity_matrix = torch.zeros([len(old_models), len(old_models)])
        for i in range(len(old_models)):
            for j in range(len(old_models)):
                similarity_matrix[i][j], _ = compute_similarity(old_models[i], old_models[j])
        
        for i, model_i in zip(idx_list, new_models):
            for j, model_j in zip(idx_list, new_models):
                similarity_matrix[i][j], _ = compute_similarity(model_i, model_j)
                
        min_s = torch.min(similarity_matrix[similarity_matrix > 0.0])
        max_s = torch.max(similarity_matrix[similarity_matrix > 0.0])

        if t % 3 == 0:
            np.savetxt(f"algorithm/invest/v3/Q_matrix/Q_mtx_{t}.txt", similarity_matrix.numpy() * 1.0, fmt='%.5f', delimiter=',')
            
        similarity_matrix = (similarity_matrix - min_s)/(max_s - min_s) * (similarity_matrix > 0.0)

        if torch.isnan(similarity_matrix).any():
            print(similarity_matrix)
            exit(0)
        
        
        similarity_matrix = similarity_matrix > self.thr
        
        if t % 3 == 0:
            np.savetxt(f"algorithm/invest/v3/Cluster/Cluster_mtx_{t}.txt", similarity_matrix.numpy() * 1.0, fmt='%d', delimiter=',')
            
        # print("binary matrix:\n", similarity_matrix * 1.0)
        impact_factor = 1/torch.sum(similarity_matrix, dim=0)
        impact_factor[torch.isinf(impact_factor)] = 0.0
        impact_factor = torch.nan_to_num(impact_factor, 0.0)
        
        with open("algorithm/invest/v3/opt_loss.txt", "a+") as file:
            loss = np.mean(np.power(impact_factor.numpy() - self.optimal_, 2))
            file.write(f'{loss}\n')
            
        impact_factor = impact_factor / torch.sum(impact_factor)
        impact_factor = impact_factor.detach().cpu().tolist()
        
        # print("impact now:", impact_factor)
        # print("last impact:", self.last_all_impact_factor)

        for idx in range(len(self.all_client_models)):
            if idx in idx_list:
                # If this client participate in this round's communication
                self.model = self.model + impact_factor[idx] * model_list[idx_list.index(idx)] - self.last_all_impact_factor[idx] * self.all_client_models[idx]
            else:
                # If not
                self.model = self.model + (impact_factor[idx] - self.last_all_impact_factor[idx]) * self.all_client_models[idx]
        
        return impact_factor

    
    def update_all_client_models_list(self, models, index):
        for idx, model in zip(index, models):
            self.all_client_models[idx] = copy.deepcopy(model)
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
        # kl_loss = KL_divergence(representation_t, representation_s, device)        # KL divergence
        kl_loss = 0.0
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss