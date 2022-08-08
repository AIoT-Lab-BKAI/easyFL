from pathlib import Path
import time
from algorithm.fedbase import BasicServer, BasicClient
from main import logger
import os
import utils.fflow as flw
import torch.multiprocessing as mp
import torch
import wandb
import json
import numpy as np
import csv
class MPBasicServer(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.gpus = option['num_gpus']
        self.num_threads = option['num_threads_per_gpu'] * self.gpus
        self.server_gpu_id = option['server_gpu_id']
        self.log_folder = option['log_folder']

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        pool = mp.Pool(self.num_threads)
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            self.current_round = round
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round, pool)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)
            with open(f'results/result/{self.result_file_name}', 'w') as f:
                
                    json.dump(self.result, f)
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        filepath = os.path.join(self.log_folder, self.option['task'], self.option['dataidx_filename']).split('.')[0]
        if not Path(filepath).exists():
            os.system(f"mkdir -p {filepath}")
        logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))

        path_save_model = f'./results/checkpoints'
        torch.save(self.model.state_dict(), path_save_model)

    def iterate(self, t, pool):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        # models, train_losses = self.communicate(self.selected_clients, pool)
        models, packages_received_from_clients = self.communicate(self.selected_clients, pool)
        
        list_uncertainty = []
        for i in range(len(self.selected_clients)):
                result_i = {
                     "round": int(self.current_round),
                     "client": int(self.selected_clients[i]),
                     "len_train_data": len(packages_received_from_clients[i]["data_idxs"]),
                     "data_idxs": packages_received_from_clients[i]["data_idxs"], #if self.current_round == 0 else packages_received_from_clients[i]["data_idxs"].tolist() ,
                     "Acc_global": float(packages_received_from_clients[i]["Acc_global"]),
                     "acc_local": float(packages_received_from_clients[i]["acc_local"]),
                     "uncertainty": float(packages_received_from_clients[i]["uncertainty"].item())
                     }
                list_uncertainty.append(packages_received_from_clients[i]["uncertainty"].item())
                self.result.append(result_i)
        
        if self.current_round == 20:
            # beta[i] = ...
            # based on list_uncertainty
            self.beta[0] = 0.1
            self.beta[1] = 0.1
            self.beta[2] = 0.1
            self.beta[3] = 0.1
            self.beta[4] = 0.1
            self.beta[5] = 0.1
            self.beta[6] = 0.1
            self.beta[7] = 0.1
            self.beta[8] = 0.1
            self.beta[9] = 0.1
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        if self.current_round != -1:
            self.client_vols = [c.datavol for c in self.clients]
            self.data_vol = sum(self.client_vols)
            self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
            print(f'Done aggregate at round {self.current_round}')
        return

    def communicate(self, selected_clients, pool):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda') # This is only 'cuda' so its can find the propriate cuda id to train
        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg, device)


    def test(self, model=None, device=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: 
            model=self.model
        if self.test_data:
            model.eval()
            loss = 0
            eval_metric = 0
            inference_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss, inference_time = self.calculator.test(model, batch_data, device)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
                inference_metric += inference_time
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            inference_metric /= len(self.test_data)
            return eval_metric, loss, inference_metric
        else: 
            return -1, -1, -1

    def test_on_clients(self, round, dataflag='valid', device='cuda'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            round: the current communication round
            dataflag: choose train data or valid data to evaluate
        :return
            evals: the evaluation metrics of the global model on each client's dataset
            loss: the loss of the global model on each client's dataset
        """
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(self.model, dataflag, device=device)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

class MPBasicClient(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)


    def train(self, model, device, beta=1, current_round=0):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
            device: the device to be trained on
        :return
        """
        model = model.to(device)
        # model.train()
        
        # data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        # optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        # for iter in range(self.epochs):
        #     for batch_id, batch_data in enumerate(data_loader):
        #         model.zero_grad()
        #         loss = self.calculator.get_loss(model, batch_data, device)
        #         loss.backward()
        #         optimizer.step()
        # return
        if self.uncertainty == 0:
            model.train()
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # model = model.to(device)
            print(len(self.train_data))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            if current_round == 0:
                num_epochs = self.epochs_round_0
            else:
                num_epochs = self.epochs
            for iter in range(self.epochs):
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    loss = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    loss.backward()
                    optimizer.step()
            return np.array(0), self.train_data.idxs
            # train function at clients
        else:
            model.train()
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # model = model.to(device)
            if current_round >= 20:
                # seed = self.name*100 + current_round
                seed = self.name + current_round*10
                print(f"name {self.name}, current_round {current_round}")
                np.random.seed(seed)
                self.train_data.beta = beta
                self.train_data.beta_idxs = np.random.choice(self.train_data.idxs, int(beta*len(self.train_data.idxs)), replace=False).tolist()
            print(len(self.train_data.beta_idxs))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            if current_round == 0:
                num_epochs = self.epochs_round_0
            else:
                num_epochs = self.epochs
                
            for iter in range(num_epochs):
                uncertainty = 0.0
                total_loss = 0.0
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    loss, unc = self.calculator.get_loss(model, batch_data, iter, device)
                    loss.backward()
                    optimizer.step() 
                    uncertainty += unc.cpu().detach().numpy() 
                    total_loss += loss
                uncertainty = uncertainty / len(data_loader.dataset)
                total_loss /= len(self.train_data)
                with open(f'./results/log_per_epoch/{self.file_log_per_epoch}', 'a') as f:
                    writer = csv.writer(f)
                    line = [iter, total_loss.item(), uncertainty]
                    writer.writerow(line)
                    
            return uncertainty, self.train_data.beta_idxs


    def test(self, model, dataflag='valid', device='cpu'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        # dataset = self.train_data if dataflag=='train' else self.valid_data
        dataset = self.train_data
        model = model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss, _ = self.calculator.test(model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss


    def reply(self, svr_pkg, device):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        # model = self.unpack(svr_pkg)
        # loss = self.train_loss(model, device)
        # self.train(model, device)
        # cpkg = self.pack(model, loss)
        
        model, beta, round = self.unpack(svr_pkg)
        Acc_global, loss_global = self.test(model)
        uncertainty, data_idxs = self.train(model, device, beta, round)
        acc_local, loss_local = self.test(model)
        cpkg = self.pack(model, data_idxs, Acc_global, acc_local, uncertainty)
        
        return cpkg


    def train_loss(self, model, device):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train', device)[1]
