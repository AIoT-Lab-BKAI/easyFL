import copy
import csv
import json
import os
import time
from heapq import nsmallest
from optparse import Option
from pathlib import Path
from turtle import mode

import numpy as np
import torch
import torch.multiprocessing as mp

import utils.fflow as flw
import wandb
from algorithm.fedbase import BasicClient, BasicServer
from main import logger
from utils.aggregate_funct import *
from utils.plot_pca import *
from sklearn.metrics import *


class MPBasicServer(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.gpus = option['num_gpus']
        self.num_threads = option['num_threads_per_gpu'] * self.gpus
        self.server_gpu_id = option['server_gpu_id']
        self.log_folder = option['log_folder']
    
    def check_converge(self, round):
        number_predict_noise = 0
        previous_number_predict_noise = 0
        PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], round)
        for i in range(0, 5):
            with open(PATH + f'/{i}.json', 'r') as f:
                list_uncertainty = json.load(f)
            for key in list_uncertainty.keys():
                if list_uncertainty[key] == 1:
                    number_predict_noise += 1
        
        Pre_PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], round-1)
        for i in range(0, 5):
            with open(Pre_PATH + f'/{i}.json', 'r') as f:
                list_uncertainty = json.load(f)
            for key in list_uncertainty.keys():
                if list_uncertainty[key] == 1:
                    previous_number_predict_noise += 1
        
        if number_predict_noise < 25000 and number_predict_noise > 22000 and abs(number_predict_noise - previous_number_predict_noise) < 250:
            print(number_predict_noise)
            print(previous_number_predict_noise)
            return True
        else:
            return False
    
    def agg_fuction(self, client_models):
        server_model = copy.deepcopy(self.model)
        server_param = []
        for name, param in server_model.state_dict().items():
            server_param=param.view(-1) if not len(server_param) else torch.cat((server_param,param.view(-1)))
        
        client_params = []
        for client_model in client_models:
            client_param=[]
            for name, param in client_model.state_dict().items():
                client_param=param.view(-1) if not len(client_param) else torch.cat((client_param,param.view(-1)))
            client_params=client_param[None, :] if len(client_params)==0 else torch.cat((client_params,client_param[None,:]), 0)
        
        for idx, client in enumerate(client_params):
            client_params[idx] = torch.sub(client, server_param)

        if self.agg_option=='median':
            agg_grads=torch.median(client_params,dim=0)[0]

        elif self.agg_option=='mean':
            agg_grads=torch.mean(client_params,dim=0)

        elif self.agg_option=='trmean':
            ntrmean = 2
            agg_grads=tr_mean(client_params, ntrmean)

        elif self.agg_option=='krum' or self.agg_option=='mkrum2' or self.agg_option=='mkrum4':
            multi_k = False if self.agg_option == 'krum' else True
            print('multi krum is ', multi_k)
            if self.agg_option=='mkrum4': 
                nkrum = 3
            else: 
                nkrum = 2
            agg_grads, krum_candidate = multi_krum(client_params, nkrum, multi_k=multi_k)
            
        elif self.agg_option=='bulyan' or self.agg_option=='bulyan5':
            if self.agg_option=='bulyan': 
                nbulyan = 1
            elif self.agg_option=='bulyan5': 
                nbulyan = 5
            agg_grads, krum_candidate=bulyan(client_params, nbulyan)

        start_idx=0
        model_grads=[]
        new_global_model = copy.deepcopy(self.model)
        for name, param in new_global_model.state_dict().items():
            param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.cuda()
            new_global_model.state_dict()[name].copy_(param + param_)
            # model_grads.append(param_)
            
        return new_global_model
    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        pool = mp.Pool(self.num_threads)
        logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds + 1):
            self.current_round = round
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round, pool)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)
            # with open(f'results/result/{self.result_file_name}', 'w') as f:
            
            #         json.dump(self.result, f)
            path_save_model = './results/checkpoints/{}_{}_{}.pt'.format(self.option['model'], self.option['noise_type'], self.option['aggregate'])
            torch.save(self.model.state_dict(), path_save_model)
            # if self.option['percent_noise_remove'] == 0:
            #     if round >= 5:
            #         if (self.check_converge(round) == True):
            #             print("Uncertainty converged")
            #             break
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # # save results as .json file
        # filepath = os.path.join(self.log_folder, self.option['task'], self.option['dataidx_filename']).split('.')[0]
        # if not Path(filepath).exists():
        #     os.system(f"mkdir -p {filepath}")
        # logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))
        with open('./results/defences/{}_{}.csv'.format(self.option['noise_type'], self.option['num_malicious']), 'a') as f:
            row = [self.option['aggregate'], logger.output['test_accs'][-1]]
            writer = csv.writer(f)
            writer.writerow(row)

    def iterate(self, round, pool):
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
        # models, packages_received_from_clients = self.communicate(self.selected_clients, pool)
        models, peer_grads = self.communicate(self.selected_clients, pool)
        peers_types = [self.clients[id].train_data.client_type for id in self.selected_clients]
        plot_updates_components(copy.deepcopy(self.model), peer_grads, peers_types, epoch=round, proportion = self.option['proportion'], attacked_class = self.option['attacked_class'],dirty_rate=self.option['dirty_rate'][0],num_malicious=self.option['num_malicious'])
        
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
        
        # if self.current_round == 20:
        #     # beta[i] = ...
        #     # based on list_uncertainty
        #     self.beta[0] = 0.1
        #     self.beta[1] = 0.1
        #     self.beta[2] = 0.1
        #     self.beta[3] = 0.1
        #     self.beta[4] = 0.1
        #     self.beta[5] = 0.1
        #     self.beta[6] = 0.1
        #     self.beta[7] = 0.1
        #     self.beta[8] = 0.1
        #     self.beta[9] = 0.1
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        if self.current_round != -1:
            # self.client_vols = [c.datavol for c in self.clients]
            # self.data_vol = sum(self.client_vols)
            # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
            self.model = self.agg_fuction(models)
            
            print(f'Done aggregate at round {self.current_round}')
        
        attacked_class_accuracies = []
        actuals, predictions = self.test_label_predictions(copy.deepcopy(self.model), device0)
        classes = list(i for i in range(10))
        print('{0:10s} - {1}'.format('Class','Accuracy'))
        # for i, r in enumerate(confusion_matrix(actuals, predictions)):
        #     print('{} - {:.2f}'.format(classes[i], r[i]/np.sum(r)*100))
        #     if i in self.option['attacked_class']:
        #         attacked_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))

        
        path_csv = 'grad/attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv'.format(len(self.option['attacked_class']),self.option['dirty_rate'][0],self.option['proportion']*50,self.option['num_malicious'])
        
        with open(path_csv + '/' + 'confusion_matrix.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                data = []
                print('{} - {:.2f}'.format(classes[i], r[i]/np.sum(r)*100))
                data.append(classes[i])
                data.append(r[i]/np.sum(r)*100)
                if i in self.option['attacked_class']:
                    attacked_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                    data.append('attacked')
                else:
                    data.append('clean')
                writer.writerow(data)
            writer.writerow(['','',''])
            writer.writerow(['','',''])
            
        
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
    def test_label_predictions(self, model, device):
        model.eval()
        actuals = []
        predictions = []
        test_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    
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


    def train(self, model, device, current_round=0):
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
            # if self.option['percent_noise_remove'] != 0:
            #     rank_dict = {}
            #     # PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], current_round)
            #     PATH = 'results/uncertainty_all_samples/resnet18/check_uncertainty_converge_2/1'
        
            #     # with open(PATH + f'/{self.name}.json', 'w') as f:
            #     #     uncertainty_dict = json.load(f)
            #     with open('results/dirty_dataidx_50' + f'/{self.name}.json', 'r') as f:
            #         dirty_list = json.load(f)
                
            #     with open(PATH + f'/{self.name}_output.json', 'r') as f:
            #         output_dict = json.load(f)

            #     # for key in uncertainty_dict.keys():
            #     #     if uncertainty_dict[key] == 1:
            #     #         rank_dict[key] = output_dict[key]
            #     for key in dirty_list:
            #         rank_dict[key] = output_dict[str(key)]

            #     # number = self.option['percent_noise_remove'] * len(uncertainty_list)
            #     number = int(self.option['percent_noise_remove'] * len(dirty_list))
            #     list_noise = nsmallest(number, rank_dict, key = rank_dict.get)
            #     self.train_data.remove_noise_specific(list_noise)
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # model = model.to(device)
            print(len(self.train_data.idxs))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            peer_grad = []
           
            for iter in range(self.epochs):
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    optimizer.step()
                    loss = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    loss.backward()
                    # Get gradient
                    for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            if iter == 0 and batch_id == 0:
                                peer_grad.append(params.grad.clone())
                            else:
                                peer_grad[i]+= params.grad.clone()
                    optimizer.step()
            # return np.array(0), self.train_data.idxs
            # train function at clients
            # if self.option['percent_noise_remove'] != 0:
            #     self.train_data.reverse_idx()
            return peer_grad
        else:
            model.train()
            
            # if self.option['percent_noise_remove'] != 0:
            #     rank_dict = {}
            #     # PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], current_round)
            #     PATH = 'results/uncertainty_all_samples/resnet18/check_uncertainty_converge_2/1'
        
            #     # with open(PATH + f'/{self.name}.json', 'w') as f:
            #     #     uncertainty_dict = json.load(f)
            #     with open('results/dirty_dataidx_50' + f'/{self.name}.json', 'r') as f:
            #         dirty_list = json.load(f)
                
            #     with open(PATH + f'/{self.name}_output.json', 'r') as f:
            #         output_dict = json.load(f)

            #     # for key in uncertainty_dict.keys():
            #     #     if uncertainty_dict[key] == 1:
            #     #         rank_dict[key] = output_dict[key]
            #     for key in dirty_list:
            #         rank_dict[key] = output_dict[str(key)]

            #     # number = self.option['percent_noise_remove'] * len(uncertainty_list)
            #     number = int(self.option['percent_noise_remove'] * len(dirty_list))
            #     list_noise = nsmallest(number, rank_dict, key = rank_dict.get)
            #     self.train_data.remove_noise_specific(list_noise)
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # model = model.to(device)
            # if current_round >= 20:
            #     # seed = self.name*100 + current_round
            #     seed = self.name + current_round*10
            #     print(f"name {self.name}, current_round {current_round}")
            #     np.random.seed(seed)
            #     self.train_data.beta = beta
            #     self.train_data.beta_idxs = np.random.choice(self.train_data.idxs, int(beta*len(self.train_data.idxs)), replace=False).tolist()
            print(len(self.train_data.idxs))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
                
            for iter in range(self.epochs):
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
                    
            # return uncertainty, self.train_data.idxs
            # if self.option['percent_noise_remove'] != 0:
            #     self.train_data.reverse_idx()


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
        model, round = self.unpack(svr_pkg)
        # loss = self.train_loss(model)
        # if round in [25, 50, 75, 100]:
        #     self.calculate_unc_all_samples(model, round)
        # if self.option['percent_noise_remove'] == 0:
        #     if round >= 4:
        #         self.calculate_unc_all_samples(model, round)    
        # Acc_global, loss_global = self.test(model)
        # uncertainty, data_idxs = self.train(model, device, round)
        # acc_local, loss_local = self.test(model)
        # cpkg = self.pack(model, data_idxs, Acc_global, acc_local, uncertainty)

        peer_grad = self.train(model, device, round)
        cpkg = self.pack(model, peer_grad)
        
        return cpkg

    def calculate_unc_all_samples(self, global_model, current_round):
        global_model.eval()
        uncertainty_dict = {}
        output_dict = {}
        for i in range(len(self.train_data)):
            data, label = self.train_data[i]
            uncertainty, output = self.calculator.get_uncertainty(global_model, data)
            uncertainty_dict[self.train_data.idxs[i]] = uncertainty.item()
            output_dict[self.train_data.idxs[i]] = output.item()

        PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], current_round)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        with open(PATH + f'/{self.name}.json', 'w') as f:
            json.dump(uncertainty_dict, f)
        with open(PATH + f'/{self.name}_output.json', 'w') as f:
            json.dump(output_dict, f)

    def train_loss(self, model, device):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train', device)[1]
