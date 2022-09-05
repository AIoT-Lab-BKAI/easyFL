from builtins import breakpoint
import time
from pathlib import Path
from unittest import result
import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
import os
import utils.fflow as flw
import wandb
import torch
import json
import csv

class BasicServer():
    def __init__(self, option, model, clients, test_data = None):
        # basic setting
        self.output_file_name = option['output_file_name']
        self.task = option['task']
        self.name = option['algorithm']
        self.model = model
        self.test_data = test_data
        self.eval_interval = option['eval_interval']
        self.num_threads = option['num_threads']
        # clients settings
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_vols = [c.datavol for c in self.clients]
        self.data_vol = sum(self.client_vols)
        self.clients_buffer = [{} for _ in range(self.num_clients)]
        self.selected_clients = []
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.decay_rate = option['learning_rate_decay']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        self.lr_scheduler_type = option['lr_scheduler']
        self.current_round = -1
        # sampling and aggregating methods
        self.sample_option = option['sample']
        self.agg_option = option['aggregate']
        self.lr=option['learning_rate']
        # names of additional parameters
        self.paras_name=[]
        self.option = option
        # server calculator
        self.calculator = fmodule.TaskCalculator(fmodule.device)
        self.server_gpu_id = option['server_gpu_id']
        self.log_folder = option['log_folder']
        self.wandb = option['wandb']
        
        # self.beta = [1 for _ in range(self.num_clients)]
        self.result_file_name = option['result_file_name']
        self.result = []
        # for idx in range(self.num_clients):
        #     self.result[idx] = {}
        if self.option['percent_noise_remove'] !=0:
            self.model.load_state_dict(torch.load('results/checkpoints/resnet18_check_uncertainty_converge_2.pt'))


    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds + 1):
            self.current_round = round
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)

            # with open(f'results/result/{self.result_file_name}', 'w') as f:
                
            #         json.dump(self.result, f)
            
            path_save_model = './results/checkpoints/{}_v2.pt'.format(self.option['model'])
            torch.save(self.model.state_dict(), path_save_model)
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        filepath = os.path.join(self.log_folder, self.option['task'], self.option['dataidx_filename']).split('.')[0]
        if not Path(filepath).exists():
            os.system(f"mkdir -p {filepath}")
        logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, packages_received_from_clients = self.communicate(self.selected_clients)
        
        # result = []
        # for idx in range(len(self.selected_clients)):
        #     result[self.selected_clients[idx]] = {}
        # list_uncertainty = []
        # for i in range(len(self.selected_clients)):
        #         result_i = {
        #              "round": int(self.current_round),
        #              "client": int(self.selected_clients[i]),
        #              "len_train_data": len(packages_received_from_clients[i]["data_idxs"]),
        #              "data_idxs": packages_received_from_clients[i]["data_idxs"], #if self.current_round == 0 else packages_received_from_clients[i]["data_idxs"].tolist() ,
        #              "Acc_global": float(packages_received_from_clients[i]["Acc_global"]),
        #              "acc_local": float(packages_received_from_clients[i]["acc_local"]),
        #              "uncertainty": float(packages_received_from_clients[i]["uncertainty"].item())
        #              }
        #         list_uncertainty.append(packages_received_from_clients[i]["uncertainty"].item())
        #         self.result.append(result_i)
            # result.append(result_i)
        
        # with open(f'results/{self.result_file_name}', 'w') as f:
            
        #         json.dump(self.result, f)
            # import pdb;pdb.set_trace()
            # f.write(json.dumps(self.result))
            # f.write(result)
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        # if self.current_round == 20:
            # beta[i] = ...
            # based on list_uncertainty
            # self.beta[0] = 0.1
            # self.beta[1] = 0.1
            # self.beta[2] = 0.1
            # self.beta[3] = 0.1
            # self.beta[4] = 0.1
            # self.beta[5] = 0.1
            # self.beta[6] = 0.1
            # self.beta[7] = 0.1
            # self.beta[8] = 0.1
            # self.beta[9] = 0.1
            

        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        start = time.time()
        if self.current_round != -1:
            self.client_vols = [c.datavol for c in self.clients]
            self.data_vol = sum(self.client_vols)
            self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
            print(f'Done aggregate at round {self.current_round}')
        end = time.time()
        if self.wandb:
            wandb.log({"Aggregation_time": end-start})
        return

    def communicate(self, selected_clients):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in selected_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(selected_clients)))
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
            pool.close()
            pool.join()
        # count the clients not dropping
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
        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg)

    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
            # "beta" : self.beta[client_id],
            "round" : self.current_round
        }

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        # train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        # return models, train_losses
        # return models, packages_received_from_clients
        return models

    def global_lr_scheduler(self, current_round):
        """
        Control the step size (i.e. learning rate) of local training
        :param
            current_round: the current communication round
        """
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{round+1} = eta_0/(round+1)"""
            self.lr = self.option['learning_rate']*1.0/(current_round+1)
            for c in self.clients:
                c.set_learning_rate(self.lr)

    def sample(self):
        """Sample the clients.
        :param
            replacement: sample with replacement or not
        :return
            a list of the ids of the selected clients
        """
        all_clients = [cid for cid in range(self.num_clients)]
        selected_clients = []
        # collect all the active clients at this round and wait for at least one client is active and
        active_clients = []
        while(len(active_clients)<1):
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        # sample clients
        if self.sample_option == 'active':
            # select all the active clients without sampling
            selected_clients = active_clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        # drop the selected but inactive clients
        # if self.current_round == 0:
        #     selected_clients = all_clients
        selected_clients = list(set(active_clients).intersection(selected_clients))
        return selected_clients

    def aggregate(self, models, p=[]):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
            p: a list of weights for aggregating
        :return
            the averaged result

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==============================================================================================|============================
        N/K * Σpk * model_k                 |1/K * Σmodel_k                  |(1-Σpk) * w_old + Σpk * model_k     |Σ(pk/Σpk) * model_k
        """
        if not models: 
            return self.model
        if self.agg_option == 'weighted_scale':
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.agg_option == 'uniform':
            return fmodule._model_average(models, p=p)
        elif self.agg_option == 'weighted_com':
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def test_on_clients(self, round, dataflag='valid', device='cpu'):
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
            eval_value, loss = c.test(self.model, dataflag, device)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

    def test(self, model=None, device='cpu'):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: model=self.model
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
            return -1,-1,-1

class BasicClient():
    def __init__(self, option, name='', train_data=None, valid_data=None):
        self.name = name
        self.frequency = 0
        # create local dataset
        self.train_data = train_data
        # self.valid_data = valid_data
        self.datavol = len(self.train_data)
        # local calculator
        self.calculator = fmodule.TaskCalculator(device=fmodule.device)
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.epochs = option['num_epochs']
        self.epochs_round_0 = option['num_epochs_round_0']
        self.learning_rate = option['learning_rate']
        self.batch_size = len(self.train_data) if option['batch_size']==-1 else option['batch_size']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.model = None
        self.num_rounds = option['num_rounds']
        # system setting
        # the probability of dropout obey distribution beta(drop, 1). The larger 'drop' is, the more possible for a device to drop
        self.drop_rate = 0 if option['net_drop']<0.01 else np.random.beta(option['net_drop'], 1, 1).item()
        self.active_rate = 1 if option['net_active']>99998 else np.random.beta(option['net_active'], 1, 1).item()
        self.wandb = option['wandb']
        self.uncertainty = option['uncertainty']
        self.file_log_per_epoch = option['file_log_per_epoch']
        self.option = option

    def train(self, model, current_round=0):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        if self.uncertainty == 0:
            model.train()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
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
            # return np.array(0), self.train_data.idxs
            # train function at clients
        else:
            model.train()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            # if current_round >= 20:
            #     # seed = self.name*100 + current_round
            #     seed = self.name + current_round*10
            #     np.random.seed(seed)
            #     self.train_data.beta = beta
            #     self.train_data.beta_idxs = np.random.choice(self.train_data.idxs, int(beta*len(self.train_data.idxs)), replace=False).tolist()
            print(len(self.train_data))
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
                # with open(f'./results/log_per_epoch/{self.file_log_per_epoch}', 'a') as f:
                #     writer = csv.writer(f)
                #     line = [iter, total_loss.item(), uncertainty]
                #     writer.writerow(line)
                    
            return uncertainty, self.train_data.idxs

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
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=32)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss, _ = self.calculator.test(model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        # return received_pkg['model'], received_pkg['beta'], received_pkg['round']
        return received_pkg['model'], received_pkg['round']


    def reply(self, svr_pkg):
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
        model, round = self.unpack(svr_pkg)
        # loss = self.train_loss(model)
        # Acc_global, loss_global = self.test(model)
        # uncertainty, data_idxs = self.train(model, round)
        # acc_local, loss_local = self.test(model)
        # cpkg = self.pack(model, data_idxs, Acc_global, acc_local, uncertainty)
        # if round == self.num_rounds:
            # self.calculate_unc_all_samples(model)
        if round in [25, 50, 75, 100]:
            self.calculate_unc_all_samples(model, round)
        self.train(model, round)
        cpkg = self.pack(model)
        return cpkg

    def calculate_unc_all_samples(self, global_model):
        uncertainty_dict = {}
        for i in range(len(self.train_data)):
            data, label = self.train_data[i]
            uncertainty = self.calculator.get_uncertainty(global_model, data)
            uncertainty_dict[self.train_data.idxs[i]] = uncertainty.item()
        PATH = 'results/uncertainty_all_samples/{}/{}.json'.format(self.option['model'], self.name)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        with open(PATH, 'w') as f:
            json.dump(uncertainty_dict, f)
        

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            # "data_idxs": data_idxs,
            # "Acc_global": Acc_global,
            # "acc_local": acc_local,
            # "uncertainty": uncertainty
        }

    def is_active(self):
        """
        Check if the client is active to participate training.
        :param
        :return
            True if the client is active according to the active_rate else False
        """
        if self.active_rate==1: return True
        else: return (np.random.rand() <= self.active_rate)

    def is_drop(self):
        """
        Check if the client drops out during communicating.
        :param
        :return
            True if the client drops out according to the drop_rate else False
        """
        if self.drop_rate==0: return False
        else: return (np.random.rand() < self.drop_rate)

    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[1]

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[1]

    def set_model(self, model):
        """
        set self.model
        :param model:
        :return:
        """
        self.model = model

    def set_learning_rate(self, lr = 0):
        """
        set the learning rate of local training
        :param lr:
        :return:
        """
        self.learning_rate = lr if lr else self.learning_rate
