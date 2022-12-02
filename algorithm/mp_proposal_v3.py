from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import numpy as np
import copy
from utils.fmodule import FModule


def KDR_loss(teacher_batch_input, student_batch_input, device):
    """
    Compute the Knowledge-distillation based KL divergence of 2 batches of layers
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


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.data_vol_this_round = None
        self.latest_impact_factor_records = [0 for cid in range(len(clients))]
        return
        
    def update_ipft_record(self):
        mu_t = 1.0 * len(self.selected_clients)/len(self.clients)
        for cid in range(len(self.latest_impact_factor_records)):
            if cid in self.selected_clients: 
                # If the client is selected this turn, renew its contribution
                self.latest_impact_factor_records[cid] = (1 - mu_t) * self.latest_impact_factor_records[cid] + mu_t * self.client_vols[cid]/self.data_vol_this_round
            else:
                # If the client is not selected this turn, its contribution decays
                self.latest_impact_factor_records[cid] = (1 - mu_t) * self.latest_impact_factor_records[cid]
        return
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()        
        self.data_vol_this_round = np.sum([self.client_vols[cid] for cid in self.selected_clients])
        
        models, train_losses = self.communicate(self.selected_clients, pool)
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol_this_round for cid in self.selected_clients]
        new_model = self.aggregate(models, p = impact_factors)
        
        self.model = self.model + len(self.selected_clients)/len(self.clients) * (new_model - self.model)
        self.update_ipft_record()
        return
    
    def pack(self, client_id):
        ipft = self.latest_impact_factor_records[client_id]
        mu = len(self.selected_clients)/len(self.clients)
        
        return {
            "model" : copy.deepcopy(self.model),
            "last_impact": ipft,
            "next_impact": (1 - mu) * ipft + mu * self.client_vols[client_id]/self.data_vol_this_round,
        }

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.local_model = None
    
    def unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['last_impact'], received_pkg['next_impact']
    
    def reply(self, svr_pkg, device):
        global_model, last_impact, next_impact = self.unpack(svr_pkg)
        # loss = self.train_loss(global_model, device)
        model, train_loss = self.train(global_model, last_impact, next_impact, device)
        cpkg = self.pack(model, train_loss)
        return cpkg
    
    def train(self, global_model: FModule, last_impact, next_impact, device):
        global_model = global_model.to(device)
        
        # Initialize model & Find complement model
        if self.local_model is None:
            self.local_model = copy.deepcopy(global_model).to(device)
        else:
            self.local_model = self.local_model.to(device)
            
        with torch.no_grad():
            complement_model = global_model - last_impact * self.local_model
        
        surogate_global_model = complement_model + next_impact * self.local_model
        surogate_global_model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, surogate_global_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        mean_loss = []
        for iter in range(self.epochs):
            losses = []
            for batch_idx, batch_data in enumerate(data_loader):
                # global_target_loss = self.calculator.get_loss(surogate_global_model, batch_data, device)                    
                
                global_target_loss = self.get_loss(surogate_global_model, global_model, batch_data, device)
                
                optimizer.zero_grad()
                global_target_loss.backward()
                losses.append(global_target_loss.detach().cpu())
                optimizer.step()
            mean_loss.append(np.mean(losses))
        
        self.local_model = (surogate_global_model - complement_model) * 1.0/(next_impact)
        self.local_model = self.local_model.to("cpu")
        return self.local_model, np.mean(mean_loss)
    
    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KDR_loss(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss + kl_loss