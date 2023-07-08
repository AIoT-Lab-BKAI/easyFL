from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.cfmtx.cfmtx import cfmtx_test
from algorithm.agg_utils.proposal_utils import ActorCritic

import torch.nn as nn
import numpy as np
import torch
import copy


def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res


def get_classifier(model):
    modules = get_module_from_model(model)
    penul = modules[-1]._parameters['weight']
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
        sim.append((x.transpose(-1,0) @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        classifier = get_classifier(model)
        self.agent = ActorCritic(num_inputs=classifier.shape, num_outputs=self.clients_per_round, hidden_size=512)
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-6) # example
        self.steps = 50 # example
        self.device = torch.device("cuda")
        self.init_states = []
        self.init_actions = []
        return
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        
        if not self.selected_clients: 
            return
        # Get classifiers
        classifiers = [get_classifier(model.to(self.device) - self.model).cpu() for model in models]
        state = torch.stack(classifiers)         # <-- Change to matrix K x d
        state = torch.unsqueeze(state, dim=0)               # <-- Change to matrix 1 x K x d
        # Processing
        # if t >= 20: 
        #     if t == 20:
        #         self.agent.reinit_weight(self.agent_optimizer, num_epochs=200, states = self.init_states, actions = self.init_actions) 
        
        # Processing
        if t > 0:
            # reward = - (np.mean(train_losses) - self.old_reward)
            reward = - (np.mean(train_losses) + np.max(train_losses) - np.min(train_losses))
            # print(np.mean(train_losses), np.max(train_losses), np.min(train_losses), reward)
            self.agent.record(reward)
            if t%self.steps == 0:
                self.agent.update(state, self.agent_optimizer) # example
        
        impact_factors = self.agent.get_action(state)
        data_vols = [self.client_vols[cid] for cid in self.selected_clients]
        
        print(impact_factors)
        print(data_vols)
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        self.model = self.aggregate(models, p = impact_factors)
        # else:
        #     device0 = torch.device(f"cuda:{self.server_gpu_id}")
        #     models = [i.to(device0) for i in models]
            
        #     self.init_states.append(state.transpose(0,1).detach())
        #     impact = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        #     self.init_actions.append(torch.FloatTensor(impact).reshape(1, -1))
            
        #     self.model = self.aggregate(models, p = impact)    
        return

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_fct = option['kd_fct']
        
    def reply(self, svr_pkg, device):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)
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
