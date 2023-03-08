from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import os


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['mu']
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.mu = option['mu']

    def train(self, model, device, current_round=0):
        if self.uncertainty == 0:
        # global parameters
            print("Here is prox")
            print(len(self.train_data.idxs))
            model = model.to(device)
            src_model = copy.deepcopy(model)
            src_model.freeze_grad()
            model.train()
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            peer_grad = []
            for iter in range(self.epochs):
                for batch_idx, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    # original_loss = self.calculator.get_loss(model, batch_data, device)
                    original_loss = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    # proximal term
<<<<<<< HEAD
                    loss_proximal = 0.0
                    for pm, ps in zip(model.parameters(), src_model.parameters()):
                        loss_proximal += torch.sum(torch.pow(pm.data-ps.data,2))
                    print(original_loss)
                    print(loss_proximal)
=======
                    loss_proximal = 0
                    for pm, ps in zip(model.parameters(), src_model.parameters()):
                        loss_proximal += torch.sum(torch.pow(pm-ps,2))
>>>>>>> 2ae7212873d8c88e4f67d0903edd91ac963e0479
                    loss = original_loss + 0.5 * self.mu * loss_proximal                #
                    loss.backward()
                    for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            if iter == 0 and batch_idx == 0:
                                peer_grad.append(params.grad.clone())
                            else:
                                peer_grad[i]+= params.grad.clone()
                    optimizer.step()
            return peer_grad

