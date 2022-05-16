from .mp_fedbase import MPBasicServer, MPBasicClient

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import torch
import os
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


def one_hot_batch(y, num_label=10, device="cuda"):
    """This function turn [2] into a one-hot vector [0, 0, 1, 0, 0, ...]

    Args:
        y (_type_): batch (a vector of scalars limited from 0 to num_label)
        num_label (int, optional): the total number of labels. Defaults to 10.
        device (str, optional): Defaults to "cuda".

    Returns:
        torch.Tensor.to(device): the one-hot matrix, each row is a one-hot vector
    """
    z = y.detach().cpu().tolist()
    one_hot = torch.zeros(len(z), num_label)
    for i, j in zip(z, range(len(z))):
        one_hot[j][i] = 1
    return one_hot.to(device)


class NoiseDataset(Dataset):
    def __init__(self, sample, length):
        self.noise_dataset = [(torch.rand_like(sample), "Noise") for i in range(length)]

    def __len__(self):
        return len(self.noise_dataset)

    def __getitem__(self, item):
        noise, label = self.noise_dataset[item]
        return noise, label
    

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.Q_matrix = torch.zeros([len(self.clients), len(self.clients)])
        self.freq_matrix = torch.zeros_like(self.Q_matrix)

        self.impact_factor = None
        self.thr = 0.975
        # self.optimal_ = np.array([1/6] * 6 + [1] * 4)
        
        self.gamma = 1
        

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        if not self.selected_clients: return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        
        self.model = self.model.to(device0)
        models = [i.to(device0) - self.model for i in models]
        
        if not self.selected_clients:
            return
        
        self.update_Q_matrix(models, self.selected_clients, t)
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.impact_factor, self.gamma = self.get_impact_factor(self.selected_clients, t)
            
            
        self.model = self.model + self.aggregate(models, self.impact_factor)
        self.update_threshold(t)
        return


    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return model_sum([model_k * pk for model_k, pk in zip(models, p)])
    
    
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
    

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        sample, _ = train_data[0]
        self.noise_data = NoiseDataset(sample, len(train_data))
        self.contst_fct = 5


    def train(self, model, device):
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
        return F.mse_loss(output_logits, -1 * abs(self.contst_fct) * one_hot_batch(targets, device=device))
    
    
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