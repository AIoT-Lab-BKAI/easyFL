from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.headtail_utils.utils import initialize
from torch.utils.data import Dataset, DataLoader

import copy
import torch


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


class DistillDataset(Dataset):
    def __init__(self, datawares_dict):
        super(DistillDataset, self).__init__()
        self.intermidate = datawares_dict['intermediate_output']
        self.output = datawares_dict['final_output']
        return
                    
    def __getitem__(self, idx):
        return (self.intermidate[idx], self.output[idx])
    
    def __len__(self):
        return self.intermidate.shape[0]
    
    
def assemble_data(client_datawares):
    assembled_dataware = {"intermediate_output": [], "final_output": []}
    for ware in client_datawares:
        assembled_dataware["intermediate_output"].append(ware["intermediate_output"])
        assembled_dataware["final_output"].append(ware["final_output"])
    
    assembled_dataware["intermediate_output"] = torch.vstack(assembled_dataware["intermediate_output"])
    assembled_dataware["final_output"] = torch.vstack(assembled_dataware["final_output"])
    return assembled_dataware

    
class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.model = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ClientModel")
        self.model.tail = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ServerTail")
        
        self.distill_epochs = 8
        self.temperature = 1.5
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
       
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(dataflag, device=device, round=round)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses
    
    def pack(self, client_id):
        return {"model" : copy.deepcopy(self.model)}
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        client_heads, client_datawares = self.communicate(self.selected_clients, pool, t)
        
        if not self.selected_clients: 
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        client_heads = [i.to(device0) for i in client_heads]
        
        self.model.head = self.aggregate(client_heads, [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        
        assembled_dataware = assemble_data(client_datawares)
        self.distill(assembled_dataware, device0)
        return
    
    def distill(self, client_datawares, device):
        self.model.tail = self.model.tail.to(device)
        optimizer = torch.optim.SGD(self.model.tail.parameters(), lr=0.001)
        
        dataset = DistillDataset(client_datawares)
        dataloader = DataLoader(dataset, batch_size=self.clients_per_round, shuffle=True)
        
        for epoch in range(self.distill_epochs):
            for intermediate_output, client_output in dataloader:

                intermediate_output = intermediate_output.to(device)
                client_output = client_output.to(device)
                
                server_output = self.model.tail(intermediate_output)
                loss = KDR_loss(client_output, server_output, device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return        


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.model = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ClientModel")
        self.class_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_factor = 1
        self.temperature = 1.5
        self.dataware = {"intermediate_output": [], "final_output": []}
        return
       
       
    def test(self, dataflag='valid', device='cpu', round=None):
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model = self.model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss
             
             
    def train(self, global_model, device, round): 
        self.model.head = global_model.head
        
        global_model.tail = global_model.tail.to(device)
        global_model.tail.freeze_grad()
        
        model = self.model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        self.dataware = {"intermediate_output": [], "final_output": []}
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, global_model.tail, batch_data, iter == self.epochs - 1, device)
                loss.backward()
                optimizer.step()
        
        self.dataware["intermediate_output"] = torch.vstack(self.dataware["intermediate_output"])
        self.dataware["final_output"] = torch.vstack(self.dataware["final_output"])
        return
    
    
    def get_loss(self, model, server_tail, batch_data, storing, device):
        X, Y = self.calculator.data_to_device(batch_data, device)
        
        intermediate_output = model.head(X)
        local_output = model.tail(intermediate_output)
        server_output = server_tail(intermediate_output)
        
        classification_loss = self.class_lossfnc(local_output, Y)
        distillation_loss = KDR_loss(server_output, local_output, device)
        
        if storing:
            self.dataware["intermediate_output"].append(intermediate_output.detach().cpu())
            self.dataware["final_output"].append(local_output.detach().cpu())
            
        return classification_loss + self.distill_factor * distillation_loss
        
        
    def reply(self, svr_pkg, device, round):
        global_model = self.unpack(svr_pkg)
        self.train(global_model, device, round)
        cpkg = self.pack(copy.deepcopy(self.model.head), self.dataware)
        return cpkg
