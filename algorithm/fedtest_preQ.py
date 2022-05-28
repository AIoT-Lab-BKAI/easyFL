from .mp_fedbase import BasicServer, BasicClient

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import torch
import os
import numpy as np
from algorithm.agg_utils.fedtest_utils import model_sum



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
        sim.append((x.T @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class NoiseDataset(Dataset):
    def __init__(self, sample, length):
        self.noise_dataset = [(torch.rand_like(sample), "Noise") for i in range(length)]

    def __len__(self):
        return len(self.noise_dataset)

    def __getitem__(self, item):
        noise, label = self.noise_dataset[item]
        return noise, label
    

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)

        self.impact_factor = None
        self.thr = 0.975        
        self.gamma = 1
        
        self.Q_matrix[:30,:30] += 1
        self.Q_matrix[30:60,30:60] += 1
        self.Q_matrix[60:80,60:80] += 1
        self.Q_matrix[80:90,80:90] += 1
        self.Q_matrix[90:100,90:100] += 1
        
        # self.path = f"/models/{self.task}/round_{self.num_rounds}"
        # self.file_save = f"{self.path}/{self.name}.pth"
        
        # self.load_model_path = option['load_model_path']
        
        # if self.load_model_path is not None:
        #     if Path(self.load_model_path).exists():
        #         print(f"Loading server model at round {self.num_rounds}...")
        #         self.model.load_state_dict(torch.load(self.load_model_path))
        #     else:
        #         print(f"Exists no {self.load_model_path}")
            
    
    def run(self):
        super().run()
        # try:
        #     if not Path(self.path).exists():
        #         os.system(f"mkdir -p {self.path}")
        #     torch.save(self.model.state_dict(), self.file_save)
        # except:
        #     print("Save model failed")
        

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: return
        device0 = torch.device("cuda")
        
        self.model = self.model.to(device0)
        models = [i.to(device0) - self.model for i in models]
        
        if not self.selected_clients:
            return
        
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            # self.update_Q_matrix(models, self.selected_clients, t)
            self.impact_factor, self.gamma = self.get_impact_factor(self.selected_clients, t)
        
        # np.savetxt(f"Q_matrix/round_{t}.txt", self.Q_matrix.numpy())
        
        self.model = self.model + self.gamma * self.aggregate(models, self.impact_factor)
        self.update_threshold(t)
        return


    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return model_sum([model_k * pk for model_k, pk in zip(models, p)], p=p)
    
    
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
        
        # Q_asterisk_mtx = self.Q_matrix/(self.freq_matrix)
        # Q_asterisk_mtx[torch.isinf(Q_asterisk_mtx)] = 0.0
        # Q_asterisk_mtx = torch.nan_to_num(Q_asterisk_mtx, 0.0)
        
        # min_Q = torch.min(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        # max_Q = torch.max(Q_asterisk_mtx[Q_asterisk_mtx > 0.0])
        # Q_asterisk_mtx = torch.abs((Q_asterisk_mtx - min_Q)/(max_Q - min_Q) * (self.freq_matrix > 0.0))
        
        # Q_asterisk_mtx = Q_asterisk_mtx > self.thr
        
        Q_asterisk_mtx = self.Q_matrix
        
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
        self.lossfunc = torch.nn.CrossEntropyLoss()
        sample, _ = train_data[0]
        self.noise_data = NoiseDataset(sample, len(train_data))
        self.contst_fct = 5


    def train(self, model, device='cuda'):
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        noise_loader = DataLoader(self.noise_data, batch_size=self.batch_size, shuffle=True)
        
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for ((batch_id, batch_data), (_, batch_noise)) in zip(enumerate(data_loader), enumerate(noise_loader)):
                model.zero_grad()
                loss = self.get_loss(model, batch_data, batch_noise, device)
                loss.backward()
                optimizer.step()
        return


    def get_contrastive_loss(self, model, batch_noise, targets, device):
        sample, _ = batch_noise
        sample = sample.to(device)
        output_logits = model(sample)
        loss = F.mse_loss(output_logits, -1.0 * abs(self.contst_fct) * F.one_hot(targets, num_classes=10))
        return loss
    
    
    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)
    
    
    def get_loss(self, model, data, noise, device=None):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[1])
        contrastive_loss = self.get_contrastive_loss(model, noise, tdata[1], device)
        return loss + contrastive_loss