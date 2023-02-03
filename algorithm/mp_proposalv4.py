from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.headtail_utils.utils import initialize
from torch.utils.data import Dataset, DataLoader
from algorithm.pkth.pkth_transfer import prob_transfer

import copy
import torch


def run_transfer(student_net, teacher_net, train_loader, learning_rates=(0.001, ), epochs=(10,), decay=0.7, init_weight=100):
    cur_weight = init_weight
    loss_params, T = {}, 2
    kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
    
    for cur_epoch, cur_lr in zip(epochs, learning_rates):
        # print("Running for ", cur_epoch, " epochs with lr = ", cur_lr)
        for i in range(cur_epoch):
            # print(cur_weight)
            weights = (1, cur_weight, cur_weight)
            prob_transfer(student_net, teacher_net, train_loader, epochs=1, lr=cur_lr, layer_weights=weights,
                          kernel_parameters=kernel_parameters, loss_params=loss_params)
            cur_weight = cur_weight * decay
            
            
class DistillDataset(Dataset):
    def __init__(self, datawares_dict):
        super(DistillDataset, self).__init__()
        self.intermidate = datawares_dict['intermediate_output']
        return

    def __getitem__(self, idx):
        return self.intermidate[idx]
    
    def __len__(self):
        return self.intermidate.shape[0]
    
    
def assemble_data(client_datawares):
    assembled_dataware = {"intermediate_output": []}
    for ware in client_datawares:
        assembled_dataware["intermediate_output"].append(ware["intermediate_output"])
    
    assembled_dataware["intermediate_output"] = torch.vstack(assembled_dataware["intermediate_output"])
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
        models, client_datawares = self.communicate(self.selected_clients, pool, t)
        
        if not self.selected_clients: 
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        heads = [model.head.to(device0) for model in models]
        tails = [model.tail.to(device0) for model in models]
        
        self.model.head = self.aggregate(heads,p=[1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        
        # assembled_dataware = assemble_data(client_datawares)
        self.distill(tails, client_datawares, device0)
        return
    
    
    def distill(self, tails, client_datawares, device):
        self.model.tail = self.model.tail.to(device)
        
        # dataset = DistillDataset(client_datawares)
        # dataloader = DataLoader(dataset, batch_size=self.clients_per_round, shuffle=True)
        
        for student, dataware in zip(tails, client_datawares):
            dataset = DistillDataset(dataware)
            dataloader = DataLoader(dataset, batch_size=self.clients_per_round, shuffle=True)
            run_transfer(student_net=student, teacher_net=self.model.tail, train_loader=dataloader,
                         epochs=self.distill_epochs, decay=0.7, init_weight=100)
        
        return        


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.model = initialize(dataset="algorithm.headtail_utils." + option['task'].split('_')[0], architecture=option['model'], modelname="ClientModel")
        self.class_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_lossfnc = torch.nn.CrossEntropyLoss()
        self.distill_factor = 0.1
        self.distill_offset = 20
        self.dataware = {"intermediate_output": []}
        return
        
    def train(self, server_tail, device, round): 
        server_tail = server_tail.to(device)
        server_tail.freeze_grad()
        
        model = self.model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        self.dataware = {"intermediate_output": []}
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, server_tail, batch_data, iter == self.epochs - 1, device, round)
                loss.backward()
                optimizer.step()
        
        self.dataware["intermediate_output"] = torch.vstack(self.dataware["intermediate_output"])
        return
    
    
    def get_loss(self, model, server_tail, batch_data, storing, device, round):
        X, Y = self.calculator.data_to_device(batch_data, device)
        
        intermediate_output = model.head(X)
        local_output = model.tail(intermediate_output)
        server_output = server_tail(intermediate_output)
        
        classification_loss = self.class_lossfnc(local_output, Y)
        distillation_loss = self.distill_lossfnc(torch.softmax(local_output/self.temperature, dim=1), torch.softmax(server_output/self.temperature, dim=1))
        
        if storing:
            self.dataware["intermediate_output"].append(intermediate_output.detach().cpu())
            
        return classification_loss + self.distill_factor * distillation_loss
        
        
    def reply(self, svr_pkg, device, round):
        self.model.head, server_tail = self.unpack(svr_pkg)
        self.train(server_tail, device, round)
        cpkg = self.pack(self.model, self.dataware)
        return cpkg
    
    def unpack(self, received_pkg):
        return received_pkg['head'], received_pkg['tail']