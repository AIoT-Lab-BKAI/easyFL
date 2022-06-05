from .fedbase import BasicServer, BasicClient

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import torch
import os
import copy
from algorithm.agg_utils.fedtest_utils import model_sum
import time, wandb


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


class NoiseDataset(Dataset):
    def __init__(self, sample, length):
        self.noise_dataset = [(torch.rand_like(sample), "Noise") for i in range(length)]

    def __len__(self):
        return len(self.noise_dataset)

    def __getitem__(self, item):
        noise, label = self.noise_dataset[item]
        return noise, label
    

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)

        self.impact_factor = None
        self.thr = 0.975        
        self.gamma = 1
        

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        
        self.model = self.model.to(device0)
        models = [i.to(device0) - self.model for i in models]
        
        if not self.selected_clients:
            return
        
        start = time.time()
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.update_Q_matrix(models, self.selected_clients, t)
            self.impact_factor, self.gamma = self.get_impact_factor(self.selected_clients, t)
            
        self.model = self.model + self.gamma * self.aggregate(models, self.impact_factor)
        self.update_threshold(t)
        
        end = time.time()
        if self.wandb:
            wandb.log({"Aggregation_time": end-start})
        return


    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return model_sum([model_k * pk for model_k, pk in zip(models, p)], p=p)
    
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
                        if sigma < 0.015 and simi > 0.998:
                            temp_Q[j,k] += simi
                            temp_F[j,k] += 1
                            temp_Q[k,j] = temp_Q[j,k]
                            temp_F[k,j] += 1
                            # print(f"Transitive: Client[{j}] - Client[{k}], By Client[{i}]: {simi:>.5f}")
                    
                    elif (self.Q_matrix[i,j] != 0) and (self.Q_matrix[i,k] == 0) and (self.Q_matrix[j,k] != 0):
                        simi, sigma = self.compute_simXY(self.Q_matrix[i,j]/self.freq_matrix[i,j],
                                                        self.Q_matrix[j,k]/self.freq_matrix[j,k])
                        if sigma < 0.015 and simi > 0.998:
                            temp_Q[i,k] += simi
                            temp_F[i,k] += 1
                            temp_Q[k,i] = temp_Q[i,k]
                            temp_F[k,i] += 1
                            # print(f"Transitive: Client[{i}] - Client[{k}], By Client[{j}]: {simi:>.5f}")
                    
                    elif (self.Q_matrix[i,j] == 0) and (self.Q_matrix[i,k] != 0) and (self.Q_matrix[j,k] != 0):
                        simi, sigma = self.compute_simXY(self.Q_matrix[i,k]/self.freq_matrix[i,k],
                                                        self.Q_matrix[j,k]/self.freq_matrix[j,k])
                        if sigma < 0.015 and simi > 0.998:
                            temp_Q[i,j] += simi
                            temp_F[i,j] += 1
                            temp_Q[j,i] = temp_Q[i,j]
                            temp_F[j,i] += 1
                            # print(f"Transitive: Client[{j}] - Client[{i}], By Client[{k}]: {simi:>.5f}")
                        
        temp_Q[temp_Q > 0] = temp_Q[temp_Q > 0]/temp_F[temp_Q > 0]
        self.Q_matrix += temp_Q
        self.freq_matrix += (temp_F > 0) * 1.0
        return
    
    @torch.no_grad()
    def update_Q_matrix(self, model_list, client_idx, t=None):
        
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
        
        if 0 in self.Q_matrix and t > 0:
            self.transitive_update_Q()
        return


    @torch.no_grad()
    def get_impact_factor(self, client_idx, t=None):
        
        Q_asterisk_mtx = self.Q_matrix/(self.freq_matrix)
        Q_asterisk_mtx[torch.isinf(Q_asterisk_mtx)] = 0.0
        Q_asterisk_mtx = torch.nan_to_num(Q_asterisk_mtx, 0.0)
        
        min_Q = torch.min(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        max_Q = torch.max(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        Q_asterisk_mtx = torch.abs((Q_asterisk_mtx - min_Q)/(max_Q - min_Q) * (self.freq_matrix > 0.0))
        
        Q_asterisk_mtx = Q_asterisk_mtx > self.thr
        
        impact_factor = 1/torch.sum(Q_asterisk_mtx, dim=0)
        impact_factor[torch.isinf(impact_factor)] = 0.0
        impact_factor = torch.nan_to_num(impact_factor, 0.0)
        impact_factor_frac = impact_factor[client_idx]
        
        num_cluster_all = torch.sum(impact_factor)
        
        temp_mtx = Q_asterisk_mtx[client_idx]
        temp_mtx = temp_mtx.T
        temp_mtx = temp_mtx[client_idx]
        
        temp_vec = 1/torch.sum(temp_mtx, dim=0)
        temp_vec[torch.isinf(temp_vec)] = 0.0
        temp_vec = torch.nan_to_num(temp_vec, 0.0)
        
        num_cluster_round = torch.sum(temp_vec)
        gamma = num_cluster_round/num_cluster_all
        
        return impact_factor_frac.detach().cpu().tolist(), gamma.detach().cpu().item()
    
    
    def update_threshold(self, t):
        self.thr = min(self.thr * (1 + 0.0005)**t, 0.998)
        return
    

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        sample, _ = train_data[0]
        self.noise_data = NoiseDataset(sample, len(train_data))
        self.contst_fct = 5
        self.kd_factor = 5


    def train(self, model, device='cuda'):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        noise_loader = DataLoader(self.noise_data, batch_size=self.batch_size, shuffle=True)
        
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for ((batch_id, batch_data), (_, batch_noise)) in zip(enumerate(data_loader), enumerate(noise_loader)):
                model.zero_grad()
                loss = self.get_loss(model, src_model, batch_data, batch_noise, device)
                loss.backward()
                optimizer.step()
        return


    def get_contrastive_loss(self, model, batch_noise, targets, device):
        sample, _ = batch_noise
        sample = sample.to(device)
        output_logits = model(sample)
        loss = F.mse_loss(output_logits, -1.0 * abs(self.contst_fct) * F.one_hot(targets, num_classes=output_logits.shape[1]))
        return loss
    
    
    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)
    
    
    def get_loss(self, model, src_model, data, noise, device=None):
        tdata = self.data_to_device(data, device)
        output_s, representation_s = model.pred_and_rep(tdata[0])                       # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                         # Teacher
        loss = self.lossfunc(output_s, tdata[1])                                        # Classifier
        contrastive_loss = self.get_contrastive_loss(model, noise, tdata[1], device)    # Contrastive
        kl_loss = KL_divergence(representation_t, representation_s, device)             # KL divergence
        return loss + contrastive_loss + self.kd_factor * kl_loss