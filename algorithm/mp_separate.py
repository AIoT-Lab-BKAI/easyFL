import copy
from .mp_fedbase import MPBasicServer, MPBasicClient
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torch
import numpy as np
import os
from pathlib import Path
import wandb

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.max_acc = 0
        return
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        _, per_accs = self.communicate(self.selected_clients, pool, t)
        self.max_acc = max(self.max_acc, np.mean(per_accs))

        # wandb record
        if self.wandb:
            wandb.log(
                {
                    "Mean Client Accuracy": np.mean(per_accs),
                    "Std Client Accuracy":  np.std(per_accs),
                    "Max Testing Accuracy": self.max_acc
                }
            )
        
        print(f"Max Testing Accuracy: {self.max_acc:>.3f}")
        print(f"Mean of Client Accuracy: {np.mean(per_accs):>.3f}")
        print(f"Std of Client Accuracy: {np.std(per_accs):>.3f}")
        return
        
        
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.model = None

    def reply(self, svr_pkg, device, round):
        model = self.unpack(svr_pkg)
        if self.model is None:
            self.model = copy.deepcopy(model)
            
        self.train(self.model, device, round)
        fin_acc, _ = self.test(self.model, device=device)
        cpkg = self.pack(0, fin_acc)
        return cpkg
    