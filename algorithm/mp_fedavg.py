from .mp_fedbase import MPBasicServer, MPBasicClient
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torch
import numpy as np
import os
from pathlib import Path

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

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
            cmtx = cmtx.cpu().numpy()
            cmtx = cmtx/np.sum(cmtx, axis=1, keepdims=True)
            
            if not Path(f"test/{self.option['algorithm']}/round_{round}").exists():
                os.makedirs(f"test/{self.option['algorithm']}/round_{round}")
            
            np.savetxt(f"test/{self.option['algorithm']}/round_{round}/server.txt", cmtx, delimiter=",", fmt="%.2f")
            
            return correct, test_loss
        else: 
            return -1, -1
        
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


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
        
        if not Path(f"test/mp_fedavg/round_{round}").exists():
            os.makedirs(f"test/mp_fedavg/round_{round}")
        
        np.savetxt(f"test/mp_fedavg/round_{round}/client_{self.name}.txt", cmtx, delimiter=",", fmt="%.2f")
            
        return correct, test_loss