from pathlib import Path
from .mp_fedbase import BasicServer, BasicClient
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import json


def read_json_idx(filename):
    idxs = []
    file_idxes = json.load(open(filename, "r"))
    for client_id in file_idxes:
        idxs += file_idxes[client_id]
    return idxs


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        single_set_idx = 'dataset_idx/' + option['dataidx_filename']
        data_folder = option['data_folder']
        train_dataset = datasets.CIFAR100(data_folder, 
                                          train=True, 
                                          download=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))
                                              ])
                                          )
        self.train_dataset = CustomDataset(train_dataset, read_json_idx(single_set_idx))
        self.optimizer = optim.Adam(self.model.parameters())
        
    def iterate(self, t):
        self.train()
        return
    
    def train(self, model=None):
        if model == None:
            model = self.model
            
        model.train()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        data_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        
        for iter in range(self.option['num_epochs']):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                self.optimizer.step()


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


