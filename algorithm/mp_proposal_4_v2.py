from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import numpy as np
import torch.multiprocessing as mp
import random

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['cpg', 'se']
        self.client_per_group = option['cpg']
        self.step_2_epochs = option['se']
        self.clusters = [[0,1,2,3,4], [5,6,7,8,9]]
        self.phase_one_models = None
        self.pairings = None
        
    def pairing_clients(self, clients, clients_per_group=2):
        """
        clients = [0,1,2,3,4,5,6,....] the list of client's id
        """
        participants = clients.copy()
        pairs = []
        
        while len(participants) > 1:
            one_pair = list(np.random.choice(participants, clients_per_group, replace=False))
            pairs.append(one_pair)
            participants = list(set(participants) - set(one_pair))
        
        if len(participants):
            pairs.append(participants)
        
        return pairs
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        # Stage 1
        self.phase_one_models, _ = self.communicate(self.selected_clients, pool)
        self.phase_one_shuffle(t)
        # Stage 2
        phase_two_models, _ = self.communicate(self.selected_clients, pool)
        self.phase_one_models = None
        
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in phase_two_models]
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return
    
    def pack(self, client_id):
        if not self.phase_one_models:
            send_model = copy.deepcopy(self.model)
        else:
            send_model = self.phase_one_models[client_id]
            
        return {
            "model" : send_model,
        }
    
    def communicate(self, selected_clients, pool):
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        sortlist = sorted(packages_received_from_clients, key=lambda d: d['id'])
        return self.unpack(sortlist)
    
    def phase_one_shuffle(self, time_step):
        random.shuffle(self.phase_one_models, lambda : 0.1)
        return
    
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        
    def pack(self, model, loss):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
        }