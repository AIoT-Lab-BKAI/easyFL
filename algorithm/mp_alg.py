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
        head_classifier = self.aggregate(head_models, p=impact_factors)
        
        rewrite_classifier(self.model, head_classifier)
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
            classifiers = []
            for cid in cluster:
                if cid != head:
                    classifiers.append(copy.deepcopy(get_ultimate_layer(models[cid], 'bias')))
            zip_list.append({"head" : head, "head_model": models[head], "classifiers": classifiers})
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
        classifiers = cluster_dict['classifiers']
        
        # listen for the client's response and return None if the client drops out
        if self.clients[head_id].is_drop(): 
            return None
        
        return self.clients[head_id].distill(head_model, classifiers, device)
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.mse = torch.nn.MSELoss()
        
    def pack(self, model, loss):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
        }

    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
        return
    
    def distill(self, model, classifiers, device):
        model = model.to(device)
        model.train()
        
        classifiers = torch.stack(classifiers, dim=0).to(device)
        classifiers.requires_grad_ = False
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=False)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, 
                                                  lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                # Normal loss without KDR
                loss = self.calculator.get_loss(model, batch_data, device)
                # Regularization with knowledge distillation
                kd_loss = self.compute_kd_loss(model, classifiers, batch_data, device)
                # Total loss
                total_loss = loss + kd_loss
                total_loss.backward()
                optimizer.step()
        
        return model

    def compute_kd_loss(self, model, classifiers, batch_data, device):
        tdata = self.data_to_device(batch_data, device)
        outputs, representations = model.pred_and_rep(tdata[0])
        
        sub_outputs = (classifiers @ representations.T).transpose(1,2)
        mse_loss = self.mse(sub_outputs, outputs)

        return mse_loss