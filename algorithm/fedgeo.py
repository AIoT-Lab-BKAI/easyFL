from algorithm.fedbase import BasicServer, BasicClient
from benchmark.toolkits import XYDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch


def separate_data(train_data):
    """
    This function separates train_data into lists
    of (sample, target) of the same label
    
    Returns dataset
    """
    separate_datasets = []
    sample_lists, target_lists = (list(t) for t in zip(*sorted(zip(train_data.X, train_data.Y), key=lambda x: x[1])))
    
    unique_value = list(np.unique(target_lists))
    print("unique_value", unique_value)
    data, target = [], []
    if len(unique_value) == 1:
        separate_datasets.append(train_data)
    else:
        for i in range(1, len(unique_value)):
            data.append(sample_lists[target_lists.index(unique_value[i-1]):target_lists.index(unique_value[i])])
            target.append(target_lists[target_lists.index(unique_value[i-1]):target_lists.index(unique_value[i])])

            if i == len(unique_value) - 1:
                data.append(sample_lists[target_lists.index(unique_value[i]):])
                target.append(target_lists[target_lists.index(unique_value[i]):])    
    
        for sample_list, target_list in zip(data, target):
            separate_datasets.append(XYDataset(X=sample_list, Y=target_list))
    
    return separate_datasets


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        
        separate_datasets = separate_data(self.train_data)
        rate = min(self.batch_size, min([len(dataset) for dataset in separate_datasets]))
        self.separate_dataloaders = [DataLoader(dataset, batch_size=int(len(dataset)/rate), shuffle=True, drop_last=True) for dataset in separate_datasets]


    def train(self, model, device):
        model.train()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)                
        for iter in range(self.epochs):
            model.zero_grad()
            loss = torch.DoubleTensor([0]).to(device)
            enumerate_list = [enumerate(data_loader) for data_loader in self.separate_dataloaders]
            for batch_id_and_batch_data in zip(*enumerate_list):
                for batch_id, batch_data in batch_id_and_batch_data:
                    tempo_loss = 1/len(batch_data) * self.get_loss(model, batch_data, device)
                    loss += tempo_loss
            loss.backward()
            optimizer.step()
        return


    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, data, device):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[1])
        return loss