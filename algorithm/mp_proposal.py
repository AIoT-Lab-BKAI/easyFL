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
        self.model = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ServerModel")
        self.distill_epochs = max(8, self.option['num_epochs'])
        self.temperature = 1.5
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
        return
    
    def pack(self, client_id):
        return {"head" : copy.deepcopy(self.model.head), "tail": copy.deepcopy(self.model.tail)}
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        heads, client_datawares = self.communicate(self.selected_clients, pool, t)
        
        if not self.selected_clients: 
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        heads = [head.to(device0) for head in heads]
        self.model.head = self.aggregate(heads,p=[1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
                
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
        self.temperature = 1.
        self.offset = 50
        self.dataware = {"intermediate_output": [], "final_output": []}
        return
        
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
                loss = self.get_loss(model, server_tail, batch_data, iter == self.epochs - 1, device, round)
                loss.backward()
                optimizer.step()
        
        self.dataware["intermediate_output"] = torch.vstack(self.dataware["intermediate_output"])
        self.dataware["final_output"] = torch.vstack(self.dataware["final_output"])
        return
    
    
    def get_loss(self, model, server_tail, batch_data, storing, device, round):
        X, Y = self.calculator.data_to_device(batch_data, device)
        
        intermediate_output = model.head(X)
        local_output = model.tail(intermediate_output)
        server_output = server_tail(intermediate_output)
        
        loss = self.class_lossfnc(local_output, Y)
        
        if round >= self.offset:
            distillation_loss = self.distill_lossfnc(torch.softmax(local_output/self.temperature, dim=1), torch.softmax(server_output/self.temperature, dim=1))
            loss += self.distill_factor * distillation_loss
        
        if storing:
            self.dataware["intermediate_output"].append(intermediate_output.detach().cpu())
            self.dataware["final_output"].append(local_output.detach().cpu())
            
        return loss
        
        
    def reply(self, svr_pkg, device, round):
        self.model.head, server_tail = self.unpack(svr_pkg)
        self.train(server_tail, device, round)
        cpkg = self.pack(copy.deepcopy(self.model.head), self.dataware)
        return cpkg
    
    def unpack(self, received_pkg):
        return received_pkg['head'], received_pkg['tail']