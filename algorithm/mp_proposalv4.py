from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.headtail_utils.utils import initialize
from torch.utils.data import Dataset, DataLoader
from algorithm.pkth.pkth_transfer import prob_transfer, prob_loss
from torch.autograd import Variable

import copy
import torch


def run_transfer(student_net, teacher_net, train_loader, learning_rate=0.001, epochs=10, decay=0.7, init_weight=100):
    cur_weight = init_weight
    kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
    
    for i in range(epochs):
        # print(cur_weight)
        weights = (1, cur_weight, cur_weight)
        prob_transfer(student_net, teacher_net, train_loader, epochs=1, lr=learning_rate, layer_weights=weights,
                        kernel_parameters=kernel_parameters)
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
    
    def test_on_clients(self, dataflag='valid', device='cuda', round=None):
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(dataflag, device=device, round=round)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses
    
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
        self.distill_factor = 0.1
        self.distill_offset = 50
        self.dataware = {"intermediate_output": []}
        return
        
    def train(self, server_tail, device, round): 
        server_tail = server_tail.to(device)
        server_tail.freeze_grad()
        
        model = self.model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
        self.dataware = {"intermediate_output": []}
        
        cur_weight, decay = 100, 0.7
        layer_weights = (1, cur_weight, cur_weight)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                
                model.zero_grad()
                X, Y = self.calculator.data_to_device(batch_data, device)
        
                head_out = model.head(X)
                local_output = model.tail(head_out)
                loss = self.class_lossfnc(local_output, Y)
                
                if iter == self.epochs - 1:
                    self.dataware["intermediate_output"].append(head_out.detach().cpu())
                
                # Distillation takes place
                distill_loss = 0
                if round >= self.distill_offset:
                    teacher_feats = server_tail.get_features(Variable(head_out))
                    student_feats = model.tail.get_features(Variable(head_out))

                    for i, (teacher_f, student_f, weight) in enumerate(zip(teacher_feats, student_feats, layer_weights)):
                        if i == 0:
                            cur_qmi = prob_loss(teacher_f, student_f, kernel_parameters=kernel_parameters)
                            distill_loss += weight * cur_qmi
                        else:
                            distill_loss += weight * prob_loss(teacher_f, student_f, kernel_parameters=kernel_parameters)

                loss = loss + self.distill_factor * distill_loss
                loss.backward()
                optimizer.step()
            
            cur_weight = cur_weight * decay
        
        self.dataware["intermediate_output"] = torch.vstack(self.dataware["intermediate_output"])
        return
        
        
    def reply(self, svr_pkg, device, round):
        self.model.head, server_tail = self.unpack(svr_pkg)
        self.train(server_tail, device, round)
        cpkg = self.pack(self.model.cpu(), self.dataware)
        return cpkg
    
    def unpack(self, received_pkg):
        return received_pkg['head'], received_pkg['tail']
    
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