from .mp_fedbase import MPBasicServer, MPBasicClient
from .utils.alg_utils.alg_utils import get_ultimate_layer, KDR_loss, rewrite_classifier
import torch
import json
import copy
import torch.nn as nn
import torch.multiprocessing as mp

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.clusters = [[0,1,2,3,4], [5,6,7,8,9]]
        # self.paras_name = ['distill_method']
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        models = [model.to('cpu') for model in models]
        head_models, head_list = self.communicate_heads(models, self.clusters, t, pool)
        
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [models[cid] for cid in range(len(models)) if cid not in head_list]
        models = models + head_models
        models = [model.to(device0) for model in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        self.model = self.aggregate(models, p=impact_factors)
        return
    
    def communicate(self, selected_clients, pool):
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        sortlist = sorted(packages_received_from_clients, key=lambda d: d['id'])
        return self.unpack(sortlist)
    
    def select_cluster_head(self, cluster, time_step):
        """
        This function selects a client id as the head of the cluster
        Args:
            cluster = [0,1,2,3,4] is a list of int (id of clients in this cluster)
        """
        return cluster[time_step % len(cluster)]
    
    def communicate_heads(self, models, clusters, time_step, pool):
        """
        The server sends the member models to their cluster head,
        heads perform distillation then return distilled models
        Args:
            models      : the models returned from self.communication
            clusters    : list of list, each member list is a cluster,
                            consists of client id. e.g. [[1,2], [3,4], ...]
        """
        zip_list = []
        head_list = []
        
        for cluster in clusters:
            head = self.select_cluster_head(cluster, time_step)
            member_models = []
            for cid in cluster:
                if cid != head:
                    member_models.append(copy.deepcopy(models[cid]))
            zip_list.append({"head" : head, "head_model": models[head], "member_models": member_models})
            head_list.append(head)
        
        packages_received_from_clients = []
        packages_received_from_clients = pool.map(self.communicate_with_heads, zip_list)
        
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return packages_received_from_clients, head_list
    
    def communicate_with_heads(self, cluster_dict):
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda') # This is only 'cuda' so its can find the propriate cuda id to train

        head_id = cluster_dict['head']
        head_model = cluster_dict['head_model']
        member_models = cluster_dict['member_models']
        
        # listen for the client's response and return None if the client drops out
        if self.clients[head_id].is_drop(): 
            return None
        
        return self.clients[head_id].distill(head_model, member_models, device)
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        # self.distill_method = option['method']
        self.distill_method = 'mse'
        self.distill_epochs = 4
        self.kd_factor = 1
        
    def pack(self, model, loss):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
        }

    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()

        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=False)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, src_model, batch_data, device)
                loss.backward()
                optimizer.step()
        return

    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KDR_loss(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss + kl_loss * self.kd_factor
    
    def distill(self, model, member_models, device):
        model = model.to(device)
        model.train()
        
        for member_model in member_models:
            member_model = member_model.to(device)
            member_model.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=False)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, 
                                                  lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.distill_epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                # Classification loss
                tdata = self.calculator.data_to_device(batch_data, device)
                outputs = model(tdata[0])
                loss = self.lossfunc(outputs, tdata[1])
                
                # Knowledge distillation loss
                sub_outputs = []
                for member_model in member_models:
                    sub_outputs.append(member_model(tdata[0]))
                sub_outputs = torch.stack(sub_outputs, dim=0).to(device)
                
                assert (outputs.shape[0] == sub_outputs.shape[1]) and (outputs.shape[1] == sub_outputs.shape[2]),\
                    f"Outputs shape {outputs.shape} is inconsistant with sub_outputs shape {sub_outputs.shape}"
                
                if self.distill_method == 'mse':
                    sub_loss = torch.sum(torch.pow(sub_outputs - outputs, 2))
                    
                elif self.distill_method == 'kldiv':
                    outputs = torch.softmax(outputs.unsqueeze(0), dim=2)
                    sub_outputs = torch.softmax(sub_outputs, dim=2)
                    sub_loss = torch.sum(sub_outputs * torch.log(sub_outputs/outputs))
                
                # Total loss
                total_loss = loss + sub_loss/outputs.shape[0]
                total_loss.backward()
                optimizer.step()
        
        return model