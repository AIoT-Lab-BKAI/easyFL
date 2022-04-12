from .mp_fedbase import MPBasicServer, MPBasicClient
import torch.nn as nn
import numpy as np

from algorithm.fedrl_utils.gae_agent.gae_agent import gae_agent
import torch
import os
import copy


def KL_distillation_divergence(teacher_batch_input, student_batch_input, device):
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
        self.impact_factor = None
        self.thr = 0.75
        
        self.agent = gae_agent(
            num_inputs=len(self.clients), 
            num_outputs=self.clients_per_round,
            hidden_size=256, 
            device=torch.device(f'cuda:{self.server_gpu_id}')
            )
        
        self.prev_reward = None
        self.prev_one_hot = None
        self.prev_disagreement = 0
        
    
    def onehot_fromlist(self, list, length=None):
        if length == None:
            length = len(self.clients)
        output = torch.zeros([1, length])
        for idx in list:
            output[0, idx] = 1
        return output
    
    
    def compute_reward(self, t, idx_one_hot, train_losses):
        """Compute reward

        Args:
            t (integer): current round
            idx_one_hot (torch.Tensor): N-sized one-hot vector decribing this round's participants
            train_losses (list of torch.Double): train losses resulted from the previous round's action

        Returns:
            double scalar : reward
        """
        if t > 0:
            client_ratio = self.clients_per_round/len(self.clients)
            losses = np.asscalar(1/np.sum(train_losses))
            new_client = torch.sum(torch.abs(idx_one_hot - self.prev_one_hot))
            return client_ratio * losses * new_client
        else:
            return None


    def iterate(self, t, pool):
        server_device = torch.device(f"cuda:{self.server_gpu_id}")
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        models = [model.to(server_device) for model in models]
        
        if not self.selected_clients:
            return
        
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.impact_factor = self.get_impact_factor(models)
        
        idx_one_hot = self.onehot_fromlist(self.selected_clients)
        state = idx_one_hot.flatten()
        self.prev_reward = self.compute_reward(t, idx_one_hot, train_losses)
        
        action = self.agent.get_action(state, self.prev_reward)
        self.agent.reflex_update(action, self.impact_factor)
        
        self.prev_one_hot = idx_one_hot        
        self.model = self.aggregate(models, p = action.cpu().tolist())
        return


    @torch.no_grad()
    def get_impact_factor(self, model_list):
        device = torch.device(f"cuda:{self.server_gpu_id}")
        self.model = self.model.to(device)
        models = []
        
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            models.append(model)
        
        similarity_matrix = torch.zeros([len(models), len(models)])
        for i in range(len(models)):
            for j in range(len(models)):
                similarity_matrix[i][j], _ = compute_similarity(models[i], models[j])
        
        similarity_matrix = (similarity_matrix - torch.min(similarity_matrix))/(torch.max(similarity_matrix) - torch.min(similarity_matrix))
        similarity_matrix = similarity_matrix > self.thr
        
        impact_factor = 1/torch.sum(similarity_matrix, dim=0)
        return impact_factor.detach()


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
        kl_loss = KL_distillation_divergence(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss