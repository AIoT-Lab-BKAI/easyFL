from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.headtail_utils.utils import initialize
from torch.utils.data import Dataset, DataLoader

import copy
import torch


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
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.server_tail = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ServerTail")
        self.distill_epochs = 8
        self.temperature = 1.5
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
        self.max_acc = 0
        return
    
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(dataflag, device=device, round=round)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses
    
    def pack(self, client_id):
        return {"model" : copy.deepcopy(self.server_tail)}
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        _, client_datawares = self.communicate(self.selected_clients, pool, t)
        
        if not self.selected_clients: 
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")        
        assembled_dataware = assemble_data(client_datawares)
        self.distill(assembled_dataware, device0)
        return
    
    def distill(self, client_datawares, device):
        self.server_tail = self.server_tail.to(device)
        optimizer = torch.optim.SGD(self.server_tail.parameters(), lr=0.001)
        
        dataset = DistillDataset(client_datawares)
        dataloader = DataLoader(dataset, batch_size=self.clients_per_round, shuffle=True)
        
        for epoch in range(self.distill_epochs):
            for intermediate_output, client_output in dataloader:

                intermediate_output = intermediate_output.to(device)
                client_output = client_output.to(device)
                
                server_output = self.server_tail(intermediate_output)
                loss = self.distill_lossfnc(torch.softmax(server_output/self.temperature, dim=1), torch.softmax(client_output/self.temperature, dim=1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return        


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.model = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ClientModel")
        self.class_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_factor = 0.1
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
        
        
    def train(self, server_tail, device, round): 
        server_tail = server_tail.to(device)
        server_tail.freeze_grad()
        
        model = self.model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        self.dataware = {"intermediate_output": [], "final_output": []}
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, server_tail, batch_data, iter == self.epochs - 1, device)
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
        distillation_loss = self.distill_lossfnc(torch.softmax(local_output/self.temperature, dim=1), torch.softmax(server_output/self.temperature, dim=1))
        
        if storing:
            self.dataware["intermediate_output"].append(intermediate_output.detach().cpu())
            self.dataware["final_output"].append(local_output.detach().cpu())
            
        return classification_loss + self.distill_factor * distillation_loss
        
        
    def reply(self, svr_pkg, device, round):
        server_tail = self.unpack(svr_pkg)
        self.train(server_tail, device, round)
        cpkg = self.pack(copy.deepcopy(self.model.tail), self.dataware)
        return cpkg
