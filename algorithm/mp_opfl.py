from .mp_fedbase import MPBasicServer, MPBasicClient
from benchmark.mnist.model.op_cnn import Model
import torch.nn as nn

import torch
import copy
import json


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.model = Model()
        self.norm_record = {cid: [] for cid in range(len(self.clients))}
        
    def run(self):
        super().run()
        json.dump(self.norm_record, open(f"inves/diff_norm_{self.option['model']}.json", "w"))
        return
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, _ = self.communicate(self.selected_clients, pool, t)

        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")        
        self.model = self.model.to(device0)
        diffs = [model.to(device0) - self.model for model in models]
        diffs_norm = [diff.norm() for diff in diffs]
        
        for cid, norm in zip(self.selected_clients, diffs_norm):
            self.norm_record[cid].append(norm.detach().cpu().item())
        
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return
        
        
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        
    def reply(self, svr_pkg, device, round):
        model = self.unpack(svr_pkg)
        # loss = self.train_loss(model, device, round)
        self.train(model, device, round)
        cpkg = self.pack(model, 0)
        return cpkg