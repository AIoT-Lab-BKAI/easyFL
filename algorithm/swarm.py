from .fedbase import BasicServer, BasicClient
import copy
import torch
from utils import fmodule


def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    sim = []
    for layer_a, layer_b in zip(a.parameters(), b.parameters()):
        x, y = torch.flatten(layer_a), torch.flatten(layer_b)
        sim.append((x.transpose(-1,0) @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.eta = option['eta']
        self.cg = self.model.zeros_like()
        self.paras_name = ['eta']
        
        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)

        self.impact_factor = None
        self.thr = 0.975
        

    def pack(self, client_id):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def unpack(self, pkgs):
        dys = [p["dy"] for p in pkgs]
        dcs = [p["dc"] for p in pkgs]
        return dys, dcs

    def iterate(self, t):
        # sample clients
        self.selected_clients = self.sample()
        # local training
        dys, dcs = self.communicate(self.selected_clients)
        if self.selected_clients == []: 
            return
        
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.update_Q_matrix(dys, self.selected_clients, t)
            self.impact_factor, self.gamma = self.get_impact_factor(self.selected_clients, t)
        
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        return

    def aggregate(self, dys, dcs):  # c_list is c_i^+
        dw = fmodule._model_average(dys, p=self.impact_factor)
        dc = fmodule._model_average(dcs, p=self.impact_factor)
        new_model = self.model + self.eta * dw
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c

    @torch.no_grad()
    def update_Q_matrix(self, model_list, client_idx, t=None):
        
        new_similarity_matrix = torch.zeros_like(self.Q_matrix)
        for i, model_i in zip(client_idx, model_list):
            for j, model_j in zip(client_idx, model_list):
                _ , new_similarity_matrix[i][j] = compute_similarity(model_i, model_j)
                
        new_freq_matrix = torch.zeros_like(self.freq_matrix)
        for i in client_idx:
            for j in client_idx:
                new_freq_matrix[i][j] = 1
        
        # Increase frequency
        self.freq_matrix += new_freq_matrix
        self.Q_matrix = self.Q_matrix + new_similarity_matrix
        return

    @torch.no_grad()
    def get_impact_factor(self, client_idx, t=None):
        
        Q_asterisk_mtx = self.Q_matrix/(self.freq_matrix)
        Q_asterisk_mtx[torch.isinf(Q_asterisk_mtx)] = 0.0
        Q_asterisk_mtx = torch.nan_to_num(Q_asterisk_mtx, 0.0)
                
        min_Q = torch.min(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        max_Q = torch.max(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        Q_asterisk_mtx = torch.abs((Q_asterisk_mtx - min_Q)/(max_Q - min_Q) * (self.freq_matrix > 0.0))
        
        Q_asterisk_mtx = Q_asterisk_mtx > self.thr
                
        impact_factor = 1/torch.sum(Q_asterisk_mtx, dim=0)
        impact_factor[torch.isinf(impact_factor)] = 0.0
        impact_factor = torch.nan_to_num(impact_factor, 0.0)
        impact_factor_frac = impact_factor[client_idx]
        
        num_cluster_all = torch.sum(impact_factor)
        
        temp_mtx = Q_asterisk_mtx[client_idx]
        temp_mtx = temp_mtx.T
        temp_mtx = temp_mtx[client_idx]
        
        temp_vec = 1/torch.sum(temp_mtx, dim=0)
        temp_vec[torch.isinf(temp_vec)] = 0.0
        temp_vec = torch.nan_to_num(temp_vec, 0.0)
        
        num_cluster_round = torch.sum(temp_vec)
        gamma = num_cluster_round/num_cluster_all
        
        return impact_factor_frac.detach().cpu().tolist(), gamma.detach().cpu().item()
    
    def update_threshold(self, t):
        self.thr = min(self.thr * (1 + 0.0005)**t, 0.998)
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = None
        
        
    def train(self, model, cg):
        model.train()
        if not self.c:
            self.c = model.zeros_like()
            self.c.freeze_grad()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        num_batches = 0
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                    pm.grad = pm.grad - pc + pcg
                optimizer.step()
                num_batches += 1
        # update local control variate c
        K = num_batches
        dy = model - src_model
        dc = -1.0 / (K * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy, dc

    def reply(self, svr_pkg):
        model, c_g = self.unpack(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc)
        return cpkg

    def pack(self, dy, dc):
        return {
            "dy": dy,
            "dc": dc,
        }

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']

    