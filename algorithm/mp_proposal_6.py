import os
from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.utils.smt_utils.mnist import MnistCnn
from algorithm.utils.alg_utils.alg_utils import KDR_loss
from utils import fmodule
from utils.fmodule import get_module_from_model
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from main import logger
import torch.multiprocessing as mp
import utils.fflow as flw

import copy
import torch
import numpy as np

num_classes = 10
feature_dim = 512

def get_ultimate_layer(model):
    penul = get_module_from_model(model)[-1]._parameters['weight']
    return penul

@torch.no_grad()
def classifier_aggregation(final_model, models, masks, indexes, device0):
    """
    Args:
        final_model : the model with the feature extractor already aggregated
        models = [model0, model5, model7, model10]
        masks = [U0, U5, U7, U10]
        indexes = [0, 5, 7, 10]
    """
    base_classifier = get_ultimate_layer(final_model).mul_(0)
    for i in range(len(indexes)):
        client_id = indexes[i]
        client_model = models[i]
        client_mask = masks[client_id].to(device0, dtype=torch.float32)
        base_classifier.add_(client_mask @ client_mask @ get_ultimate_layer(client_model))
    return

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
        """
        Phi is the concatenation of all Us (m x m) of all clients (n).
        Phi is a torch tensor of shape n x m x m
            n: the number of clients
            m: the number of classes
        """
        self.Phi = torch.zeros([self.num_clients, num_classes, num_classes])
        
        """
        Psi describes the distribution of samples' representation 
        in each clients.
        Psi is a torch tensor of shape n x d
            n: the number of clients
            d: the dimention of the representation
        """
        self.Psi = torch.zeros([self.num_clients, feature_dim])
        self.model = MnistCnn()
        
    def update_auxil_infor(self, indexes, Psis, masks):
        for index, Psi_i, mask_i in zip(indexes, Psis, masks):
            self.Psi[index] = copy.copy(Psi_i)
            self.Phi[index] = copy.copy(mask_i)
        return

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        
        models, train_losses, Psis, masks, indexes = self.communicate(self.selected_clients, pool, round=t)
        self.update_auxil_infor(indexes, Psis, masks)
        
        if not self.selected_clients:
            return
            
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        # Aggregation
        impact_factors = 1/len(self.selected_clients)
        self.model = fmodule._model_sum([model_k * impact_factors for model_k in models])
        
        # print("Special Aggregation for the Classifier")
        classifier_aggregation(self.model, models, masks, indexes, device0)
        self.model.prepare(self.Phi, self.Psi)
        return

    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        Psis = [cp["Psi_i"] for cp in packages_received_from_clients]
        masks = [cp["mask"] for cp in packages_received_from_clients]
        indexes = [cp["id"] for cp in packages_received_from_clients]
        return models, train_losses, Psis, masks, indexes

    def test(self, model=None, device=None, round=None):
        if model==None: 
            model=self.model
        if self.test_data:
            
            model.cuda()
            model.eval()
            loss_fn = torch.nn.CrossEntropyLoss()
            
            test_loader = DataLoader(self.test_data, batch_size=32, shuffle=True, drop_last=False)
            size = len(test_loader.dataset)
            num_batches = len(test_loader)
            
            test_loss, correct = 0, 0
            confmat = ConfusionMatrix(num_classes=10).to(device)
            cmtx = 0
            
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    cmtx += confmat(pred, y)

            test_loss /= num_batches
            correct /= size
            up = cmtx.cpu().numpy()
            down = np.sum(up, axis=1, keepdims=True)
            down[down == 0] = 1
            cmtx = up/down
            
            if not Path(f"test/{self.option['algorithm']}/round_{round}").exists():
                os.makedirs(f"test/{self.option['algorithm']}/round_{round}")
            
            np.savetxt(f"test/{self.option['algorithm']}/round_{round}/server.txt", cmtx, delimiter=",", fmt="%.2f")
            
            return correct, test_loss
        else: 
            return -1, -1


class Client(MPBasicClient):
    _encode_key = torch.rand([num_classes, num_classes])
    
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.replossfunc = torch.nn.MSELoss()
        self.compress_repre = torch.randn([1, feature_dim], requires_grad=True)
        self.mask = self.create_mask(num_classes=num_classes)                           # Mask = R @ G^(-1)
        self.cr_lr = 0.2
        
    def create_mask(self, num_classes):
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=1)
        mask = torch.zeros([num_classes, num_classes])
        for batch_id, batch_data in enumerate(data_loader):
            _, y = batch_data
            mask[y.item(), y.item()] = 1
        # print(f"Client {self.name} mask:", mask)
        return mask
    
    def reply(self, svr_pkg, device, round):
        model= self.unpack(svr_pkg)
        model.update_mask(self.mask)        # Use masked low-rank linear classifier
        
        loss = 0
        if round > 0:
            loss = self.train_loss(model, device, round)
            
        Psi_i = self.train(model, device, round)
        cpkg = self.pack(model, loss, Psi_i)
        return cpkg
    
    def pack(self, model, loss, Psi_i):
        return {
            "id" : int(self.name),
            "model" : model,
            "train_loss": loss,
            "Psi_i": Psi_i,
            # "mask": Client._encode_key @ self.mask @ torch.linalg.inv(Client._encode_key),
            "mask": self.mask,
        }
        
    def train(self, model, device, round):
        self.compress_repre = self.compress_repre.to(device)
        
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):

                # mimimizing the classifier loss
                model.zero_grad()
                loss, representations = self.get_loss(model, src_model, batch_data, device)
                loss.backward()
                optimizer.step()
                
                # maximizing the similarity of compress_repre and the representations
                self.compress_repre.retain_grad()
                similarity = (self.compress_repre @ representations.T)/ \
                                (torch.norm(self.compress_repre, dim=1, keepdim=True) @ torch.norm(representations, dim=1, keepdim=True).T)
                rep_reward = torch.sum(similarity)
                rep_reward.backward()
                self.compress_repre = self.compress_repre + self.cr_lr * self.compress_repre.grad
        
        return self.compress_repre.detach().cpu()
    
    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        # kl_loss = KDR_loss(representation_t, representation_s, device)        # KL divergence
        repre_loss = self.replossfunc(representation_t, representation_s)
        loss = self.lossfunc(output_s, tdata[1])
        return loss + 0.01 * repre_loss, representation_s.detach()

    def test(self, model, dataflag='valid', device='cpu', round=None):
        model.cuda()
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        
        test_loader = DataLoader(self.train_data, batch_size=8, shuffle=True, drop_last=False)
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        
        test_loss, correct = 0, 0
        confmat = ConfusionMatrix(num_classes=10).to(device)
        cmtx = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                cmtx += confmat(pred, y)

        test_loss /= num_batches
        correct /= size
        up = cmtx.cpu().numpy()
        down = np.sum(up, axis=1, keepdims=True)
        down[down == 0] = 1
        cmtx = up/down
        
        if not Path(f"test/mp_proposal_6/round_{round}").exists():
            os.makedirs(f"test/mp_proposal_6/round_{round}")
        
        np.savetxt(f"test/mp_proposal_6/round_{round}/client_{self.name}_after_aggregation.txt", cmtx, delimiter=",", fmt="%.2f")
            
        return correct, test_loss
    