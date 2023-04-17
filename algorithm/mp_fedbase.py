import copy
import csv
import json
import os
import time
from heapq import nsmallest
from optparse import Option
from pathlib import Path
# from turtle import mode

import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp

import utils.fflow as flw
import wandb
from algorithm.fedbase import BasicClient, BasicServer
from main import logger
from utils.aggregate_funct import *
from utils.plot_pca import *
from sklearn.metrics import *
from sklearn.cluster import KMeans


class MPBasicServer(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.gpus = option['num_gpus']
        self.num_threads = option['num_threads_per_gpu'] * self.gpus
        self.server_gpu_id = option['server_gpu_id']
        self.log_folder = option['log_folder']
        self.confidence_score = {}
        self.computation_time = {}
        self.type_image = {}
        # self.min_avg_cs_clean_threshold = [0.2, 0.2, 0.2, 0.2, 0.2]
        # self.frequent_update_threshold = [0, 0, 0, 0, 0]
        # self.cs_clean_recently = [[0.2, 0.2, 0.2],[0.2, 0.2, 0.2],[0.2, 0.2, 0.2],[0.2, 0.2, 0.2],[0.2, 0.2, 0.2]]
    
    # def check_converge(self, round):
    #     number_predict_noise = 0
    #     previous_number_predict_noise = 0
    #     PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], round)
    #     for i in range(0, 5):
    #         with open(PATH + f'/{i}.json', 'r') as f:
    #             list_uncertainty = json.load(f)
    #         for key in list_uncertainty.keys():
    #             if list_uncertainty[key] == 1:
    #                 number_predict_noise += 1
        
    #     Pre_PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], round-1)
    #     for i in range(0, 5):
    #         with open(Pre_PATH + f'/{i}.json', 'r') as f:
    #             list_uncertainty = json.load(f)
    #         for key in list_uncertainty.keys():
    #             if list_uncertainty[key] == 1:
    #                 previous_number_predict_noise += 1
        
    #     if number_predict_noise < 25000 and number_predict_noise > 22000 and abs(number_predict_noise - previous_number_predict_noise) < 250:
    #         print(number_predict_noise)
    #         print(previous_number_predict_noise)
    #         return True
    #     else:
    #         return False
    
    def agg_fuction(self, client_models):
        print(self.agg_option)
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

        elif self.agg_option=='krum' or self.agg_option=='mkrum' or self.agg_option=='mkrum3':
            multi_k = False if self.agg_option == 'krum' else True
            print('multi krum is ', multi_k)
            if self.agg_option=='mkrum3': 
                nkrum = 3
            else: 
                nkrum = 1
            agg_grads, krum_candidate = multi_krum(client_params, nkrum, multi_k=multi_k)
            
        elif self.agg_option=='bulyan' or self.agg_option=='bulyan2':
            if self.agg_option=='bulyan': 
                nbulyan = 1
            elif self.agg_option=='bulyan2': 
                nbulyan = 2
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
            
            if round == self.num_rounds:
                path_js = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                
                for client in self.clients:
                    # dict_set = client.train_data.type_image_idx 
                    # dict_list = {}
                    # for key in dict_set.keys():
                    #     dict_list[key] = list(dict_set[key])
                    self.type_image[client.name] = client.train_data.type_image_idx   
                # with open(path_js + 'type_image.json', 'w') as json_file:
                #     json.dump(self.type_image, json_file, indent=4)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)
            print('Max test acc = ',max(logger.output['test_accs']))
            # with open(f'results/result/{self.result_file_name}', 'w') as f:
            
            #         json.dump(self.result, f)
            # path_save_model = './results/checkpoints/{}_{}_{}.pt'.format(self.option['model'], self.option['noise_type'], self.option['aggregate'])
            # path_save_model = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/model.pt' 
            # torch.save(self.model.state_dict(), path_save_model)
            # if self.option['percent_noise_remove'] == 0:
            #     if round >= 5:
            #         if (self.check_converge(round) == True):
            #             print("Uncertainty converged")
            #             break
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # # save results as .json file
        filepath = os.path.join(self.log_folder, 'log/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'])
        if not Path(filepath).exists():
            os.system(f"mkdir -p {filepath}")
        logger.save(os.path.join(filepath, 'logger.json'))
        # logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))
        # with open('./results/defences/{}_{}.csv'.format(self.option['noise_type'], self.option['num_malicious']), 'a') as f:
        #     row = [self.option['aggregate'], logger.output['test_accs'][-1]]
        #     writer = csv.writer(f)
        #     writer.writerow(row)

    # def cluster_2(self, list_idx, models, acc_before_trains, round, iter, cluster_client_id):
    #     print(f'Cluster client {cluster_client_id}')
    #     if len(list_idx) == 0:
    #         return [], {}, 2
    #     elif len(list_idx) == 1:
    #         for idx,client in enumerate(self.selected_clients):
    #             if idx in list_idx:
    #                 confidence_score_dict_client = self.confidence_score[round][int(client)]
    #                 avg_confidence_score = sum(confidence_score_dict_client.values())/len(confidence_score_dict_client)
    #         path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv/{}'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'], 0)
    #         file = f"epoch{round}.csv"
    #         path_csv = os.path.join(path_, file)
    #         df = pd.read_csv(path_csv, index_col=0)
    #         X = df.loc[list_idx, ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"]].to_numpy()
    #         Y = df.loc[list_idx,["target"]].to_numpy()
    #         Y_ = []
    #         attacker_idx = []
    #         for y, idx in enumerate(list_idx):
    #             if Y[y] == "attacker":
    #                 attacker_idx.append(idx)
    #                 Y_.append(1)
    #             else:
    #                 Y_.append(0)
    #         print(f"Attacker idx: {attacker_idx}")
    #         if avg_confidence_score < self.min_avg_cs_clean_threshold[cluster_client_id]:  
    #             attacker_cluster = list_idx
    #             chosen_cluster = []
    #         else: 
    #             attacker_cluster = []
    #             chosen_cluster = list_idx
    #             self.cs_clean_recently[cluster_client_id].pop(0)
    #             self.cs_clean_recently[cluster_client_id].append(avg_confidence_score)
    #         true_pred = list(set(attacker_idx) & set(attacker_cluster))
    #         print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
            
    #         dictionary = {
    #         "attacker_idx": attacker_idx,
    #         "cluster 0": [],
    #         "cluster 1": [],
    #         "Avg_acc_before_train_cluster_0": 0,
    #         "Avg_acc_before_train_cluster_1": 0,
    #         "Chosen cluster": chosen_cluster,
    #         "attacker_cluster": attacker_cluster,
    #         "true prediction attacker": [int(i) for i in true_pred],
    #         }
    #         return chosen_cluster, dictionary, 2
            
    #     path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv/{}'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'], 0)
    #     file = f"epoch{round}.csv"
    #     path_csv = os.path.join(path_, file)
    #     df = pd.read_csv(path_csv, index_col=0)
    #     X = df.loc[list_idx, ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"]].to_numpy()
    #     avg_confidence_score_list = []
    #     for idx,client in enumerate(self.selected_clients):
    #         if idx in list_idx:
    #             confidence_score_dict_client = self.confidence_score[round][int(client)]
    #             avg_confidence_score_list.append(sum(confidence_score_dict_client.values())/len(confidence_score_dict_client))
    #     # avg_confidence_score_array = np.expand_dims(np.asarray(avg_confidence_score_list),axis=1)
    #     # X = np.concatenate((X, avg_confidence_score_array), axis=1)
    #     Y = df.loc[list_idx,["target"]].to_numpy()
    #     Y_ = []
    #     attacker_idx = []
    #     for y, idx in enumerate(list_idx):
    #         if Y[y] == "attacker":
    #             attacker_idx.append(idx)
    #             Y_.append(1)
    #         else:
    #             Y_.append(0)
    #     # attacker_idx = np.nonzero(Y_)[0]        
    #     print(f"Attacker idx: {attacker_idx}")
    #     kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
    #     y_pred = kmeans.labels_
    #     cluster_0 = []
    #     cluster_0_id = []
    #     cluster_1 = []
    #     cluster_1_id = []
        
    #     # cluster_2 = []
    #     for i,idx in enumerate(list_idx):
    #         if y_pred[i] == 0:
    #             cluster_0.append(idx)
    #             cluster_0_id.append(i)
    #         elif y_pred[i] == 1:
    #             cluster_1.append(idx)
    #             cluster_1_id.append(i)
                
    #         # else: 
    #         #     cluster_2.append(y)
    #     print(f"Cluster 0: {cluster_0}")
    #     print(f"Cluster 1: {cluster_1}")
        
    #     models_cluster_0 = [models[i] for i in cluster_0]
    #     models_cluster_1 = [models[i] for i in cluster_1]
    #     # models_cluster_2 = [models[i] for i in cluster_2]
        
    #     acc_before_train_cluster_0 = [avg_confidence_score_list[i] for i in cluster_0_id]
    #     acc_before_train_cluster_1 = [avg_confidence_score_list[i] for i in cluster_1_id]
        
    #     Avg_acc_before_train_cluster_0 = sum(acc_before_train_cluster_0)/len(acc_before_train_cluster_0)
    #     Avg_acc_before_train_cluster_1 = sum(acc_before_train_cluster_1)/len(acc_before_train_cluster_1)
        
    #     # aggregate_model_cluster_0 = self.agg_fuction(models_cluster_0)
    #     # aggregate_model_cluster_1 = self.agg_fuction(models_cluster_1)
    #     # aggregate_model_cluster_2 = self.agg_fuction(models_cluster_2)
        
    #     print(f'Avg_acc_before_train_cluster_0 : {Avg_acc_before_train_cluster_0}')
    #     print(f'Avg_acc_before_train_cluster_1 : {Avg_acc_before_train_cluster_1}')
        
    #     # if round < 10:
    #     #     attack_threshold = (round/2)*0.04 + 0.21
    #     # else:
    #     #     attack_threshold = 0.40
    #     if Avg_acc_before_train_cluster_0 > Avg_acc_before_train_cluster_1: 
            
    #         # if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 > 0.02:
    #         #     # self.model = aggregate_model_cluster_0
    #         #     chosen_cluster = cluster_0
    #         #     attacker_cluster = cluster_1
    #         # else: 
    #         #     if Avg_acc_before_train_cluster_0 > 0.1 and Avg_acc_before_train_cluster_1 > 0.1:
    #         #     # self.model = self.agg_fuction(models)
    #         #         chosen_cluster = list_idx
    #         #         attacker_cluster = []
    #         #     else: 
    #         #         chosen_cluster = []
    #         #         attacker_cluster = list_idx
    #         if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 >= 0.02:
    #                 # self.model = aggregate_model_cluster_0
    #                 chosen_cluster = cluster_0
    #                 min_avg_cs_clean_cluster = min(acc_before_train_cluster_0)
    #                 attacker_cluster = cluster_1
    #                 self.cs_clean_recently[cluster_client_id].pop(0)
    #                 self.cs_clean_recently[cluster_client_id].append(min_avg_cs_clean_cluster)
    #         else: 
    #             # if iter == 0:
    #             #     return [], [], 1.0
    #             # else:
    #                 if Avg_acc_before_train_cluster_0 < self.min_avg_cs_clean_threshold[cluster_client_id]:
    #                     chosen_cluster = []
    #                     min_avg_cs_clean_cluster = 2
    #                     attacker_cluster = list_idx
    #                 else:
    #                     chosen_cluster = list_idx
    #                     min_avg_cs_clean_cluster = 2
    #                     attacker_cluster = []
    #                     self.cs_clean_recently[cluster_client_id].pop(0)
    #                     self.cs_clean_recently[cluster_client_id].append(min(acc_before_train_cluster_1))
    #             # if Avg_acc_before_train_cluster_0 < attack_threshold:
    #             #     chosen_cluster = []
    #             #     attacker_cluster = list_idx
    #             # else: 
    #             #     chosen_cluster = list_idx
    #             #     attacker_cluster = []
    #             # # if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 >= 0.03:
    #             # #     # self.model = aggregate_model_cluster_0
    #             # #     chosen_cluster = cluster_0
    #             # #     attacker_cluster = cluster_1
    #             # # else: 
    #             # #     chosen_cluster = list_idx
    #             # #     attacker_cluster = []
    #     else:
    #         # if Avg_acc_before_train_cluster_1 - Avg_acc_before_train_cluster_0 > 0.02:
    #         #     # self.model = aggregate_model_cluster_1
    #         #     chosen_cluster = cluster_1
    #         #     attacker_cluster = cluster_0
    #         # else:
    #         #     # self.model = self.agg_fuction(models)
    #         #     if Avg_acc_before_train_cluster_0 > 0.1 and Avg_acc_before_train_cluster_1 > 0.1:
    #         #     # self.model = self.agg_fuction(models)
    #         #         chosen_cluster = list_idx
    #         #         attacker_cluster = []
    #         #     else: 
    #         #         chosen_cluster = []
    #         #         attacker_cluster = list_idx
            
    #             if Avg_acc_before_train_cluster_1 - Avg_acc_before_train_cluster_0 >= 0.02:
    #                 # self.model = aggregate_model_cluster_1
    #                 chosen_cluster = cluster_1
    #                 min_avg_cs_clean_cluster = min(acc_before_train_cluster_1)
    #                 attacker_cluster = cluster_0
    #                 self.cs_clean_recently[cluster_client_id].pop(0)
    #                 self.cs_clean_recently[cluster_client_id].append(min_avg_cs_clean_cluster)
            
    #             else: 
    #                 # if iter == 0:
    #                 #     return [], [], 1.0
    #                 # else:
    #                     if Avg_acc_before_train_cluster_1 < self.min_avg_cs_clean_threshold[cluster_client_id]:
    #                         chosen_cluster = []
    #                         min_avg_cs_clean_cluster = 2
    #                         attacker_cluster = list_idx
    #                     else:
    #                         chosen_cluster = list_idx
    #                         min_avg_cs_clean_cluster = 2
    #                         attacker_cluster = []
    #                         self.cs_clean_recently[cluster_client_id].pop(0)
    #                         self.cs_clean_recently[cluster_client_id].append(min(acc_before_train_cluster_0))
            
    #                 # if Avg_acc_before_train_cluster_1 < attack_threshold:
    #                 #     chosen_cluster = []
    #                 #     attacker_cluster = list_idx
    #                 # else:
    #                 #     chosen_cluster = list_idx
    #                 #     attacker_cluster = []
            
        
    #     true_pred = list(set(attacker_idx) & set(attacker_cluster))
    #     print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
    #     print('\n')
        
    #     dictionary = {
    #     "attacker_idx": attacker_idx,
    #     "cluster 0": cluster_0,
    #     "cluster 1": cluster_1,
    #     "Avg_acc_before_train_cluster_0": Avg_acc_before_train_cluster_0,
    #     "Avg_acc_before_train_cluster_1": Avg_acc_before_train_cluster_1,
    #     "Chosen cluster": chosen_cluster,
    #     "attacker_cluster": attacker_cluster,
    #     "true prediction attacker": [int(i) for i in true_pred],
    #     }
    #     return chosen_cluster, dictionary, min_avg_cs_clean_cluster

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
        # models, peer_grads, acc_before_trains, loss_before_trains, confidence_score_dict = self.communicate(self.selected_clients, pool)
        models, acc_before_trains, loss_before_trains, confidence_score_dict, calculate_cs_time, train_time = self.communicate(self.selected_clients, pool)
        self.computation_time[round] = {}
        if self.option["agg_algorithm"] == "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)":
            self.computation_time[round]["calculate_cs_time"] = calculate_cs_time
            self.computation_time[round]["train_time"] = train_time
        else:
            self.computation_time[round]["train_time"] = train_time
            
            
        self.confidence_score[round] = {}
        # for client in self.clients:
        for idx,client in enumerate(self.selected_clients):
            self.confidence_score[round][int(client)] = confidence_score_dict[idx]
            
        peers_types = [self.clients[id].train_data.client_type for id in self.selected_clients]
        # plot_updates_components(copy.deepcopy(self.model), peer_grads, peers_types, peers_id=self.selected_clients, epoch=round, proportion = self.option['proportion'], attacked_class = self.option['attacked_class'],dirty_rate=self.option['dirty_rate'][0],num_malicious=self.option['num_malicious'], agg_algorithm=self.option['agg_algorithm'], algorithm= self.option['algorithm'], type_noise=self.option['outside_noise'])
        # path_js = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/type_noise_{}/proportion_{}/num_malicious_{}/'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0],self.option['outside_noise'], self.option['proportion']*50, self.option['num_malicious'])
        path_js = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
        if not os.path.exists(path_js):
            os.makedirs(path_js)
        # with open(path_js + 'confidence_score.json', 'w') as json_file:
        #     json.dump(self.confidence_score, json_file, indent=4)
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        # if self.current_round != -1:
        if self.agg_option == 'mean':
            if self.option['agg_algorithm'] == "Kmean_acc_0.02_aggregate_attacker_by_cs_(csi+0.1)":
                path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/type_noise_{}/proportion_{}/num_malicious_{}/csv/{}'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0],self.option['outside_noise'], self.option['proportion']*50, self.option['num_malicious'], 0)
                file = f"epoch{round}.csv"
                path_csv = os.path.join(path_, file)
                df = pd.read_csv(path_csv, index_col=0)
                X = df.loc[:, ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"]].to_numpy()
                avg_confidence_score_list = []
                for idx,client in enumerate(self.selected_clients):
                    # if idx in list_idx:
                    confidence_score_dict_client = self.confidence_score[round][int(client)]
                    avg_confidence_score_list.append(sum(confidence_score_dict_client.values())/len(confidence_score_dict_client))
                Y = df.loc[:,["target"]].to_numpy()
                Y_ = []
                attacker_idx = []
                for y, idx in enumerate(self.selected_clients):
                    if Y[y] == "attacker":
                        attacker_idx.append(idx)
                        Y_.append(1)
                    else:
                        Y_.append(0)
                print(f"Attacker idx: {attacker_idx}")
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
                y_pred = kmeans.labels_
                cluster_0 = []
                cluster_0_id = []
                cluster_1 = []
                cluster_1_id = []
                
                for i,idx in enumerate(self.selected_clients):
                    if y_pred[i] == 0:
                        cluster_0.append(idx)
                        cluster_0_id.append(i)
                    elif y_pred[i] == 1:
                        cluster_1.append(idx)
                        cluster_1_id.append(i)

                print(f"Cluster 0: {cluster_0}")
                print(f"Cluster 1: {cluster_1}")
                
                acc_before_train_cluster_0 = [acc_before_trains[i] for i in cluster_0_id]
                acc_before_train_cluster_1 = [acc_before_trains[i] for i in cluster_1_id]
                
                Avg_acc_before_train_cluster_0 = sum(acc_before_train_cluster_0)/len(acc_before_train_cluster_0)
                Avg_acc_before_train_cluster_1 = sum(acc_before_train_cluster_1)/len(acc_before_train_cluster_1)
                

                print(f'Avg_acc_before_train_cluster_0 : {Avg_acc_before_train_cluster_0}')
                print(f'Avg_acc_before_train_cluster_1 : {Avg_acc_before_train_cluster_1}')
                
                if Avg_acc_before_train_cluster_0 > Avg_acc_before_train_cluster_1: 
                    
                    if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 >= 0.02:
                            chosen_cluster = cluster_0
                            attacker_cluster = cluster_1
                    else: 
                            # if Avg_acc_before_train_cluster_0 < 0.1:
                            #     chosen_cluster = []
                            #     attacker_cluster = self.selected_clients
                            # else:
                            chosen_cluster = self.selected_clients
                            attacker_cluster = []
                else:
                        if Avg_acc_before_train_cluster_1 - Avg_acc_before_train_cluster_0 >= 0.02:
                            chosen_cluster = cluster_1
                            attacker_cluster = cluster_0
                        else: 
                                # if Avg_acc_before_train_cluster_1 < 0.1:
                                #     chosen_cluster = []
                                #     attacker_cluster = self.selected_clients
                                # else:
                                chosen_cluster = self.selected_clients
                                attacker_cluster = []
                
                true_pred = list(set(attacker_idx) & set(attacker_cluster))
                print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                print('\n')

                sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                p_ = []
                for idx,client in enumerate(self.selected_clients):
                    if client in chosen_cluster:
                        p_.append(1.0 * self.client_vols[client]/sum_sample)
                    else:
                        # p_.append(0)
                        max_cs = max(acc_before_trains)
                        min_cs = min(acc_before_trains)
                        csi = (acc_before_trains[idx] - min_cs)/(max_cs-min_cs)
                        # if self.option['agg_algorithm'] == "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.1)":
                        p_.append((csi+0.1) * self.client_vols[client]/sum_sample)
                        # else:
                        #     p_.append(csi * self.client_vols[client]/sum_sample)

                print('p_ = ', p_)
                if len(chosen_cluster) > 0:
                    # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/sum_sample for cid in predicted_normal])
                    self.model = self.aggregate(models, p = p_)
                        
                dictionary = {
                "attacker_idx": [int(_) for _ in attacker_idx],
                "cluster 0": [int(_) for _ in cluster_0],
                "cluster 1": [int(_) for _ in cluster_1],
                "Avg_acc_before_train_cluster_0": Avg_acc_before_train_cluster_0,
                "Avg_acc_before_train_cluster_1": Avg_acc_before_train_cluster_1,
                "Chosen cluster": [int(_) for _ in chosen_cluster],
                "attacker_cluster": [int(_) for _ in attacker_cluster],
                "true prediction attacker": [int(i) for i in true_pred],
                }
                path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/type_noise_{}/proportion_{}/num_malicious_{}/'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0],self.option['outside_noise'], self.option['proportion']*50, self.option['num_malicious'])
                listObj = []
                if round != 1:
                    with open(path_ + 'log.json') as fp:
                        listObj = json.load(fp)
                
                listObj.append(dictionary)
                
                with open(path_ + 'log.json', 'w') as json_file:
                    json.dump(listObj, json_file, indent=4)
                    
            elif self.option['agg_algorithm'] != "fedavg":  
                ours_server_time_start = time.time()
                list_peak = []
                list_confidence_score = []
                
                df_round = pd.DataFrame()
                for idx,client in enumerate(self.selected_clients):
                        confidence_score_dict_client = self.confidence_score[round][int(client)]
                        df_client = pd.DataFrame.from_dict(confidence_score_dict_client, orient='index', columns=['Cs'])
                        ax = sns.displot(df_client, x='Cs', kind="kde")
                        for ax in ax.axes.flat:
                            # print (ax.lines)
                            for line in ax.lines:
                                x = line.get_xdata() # Get the x data of the distribution
                                y = line.get_ydata() # Get the y data of the distribution
                        maxid = np.argmax(y) 
                        list_peak.append(y[maxid])
                        list_confidence_score.append(df_client.Cs.mean())
                        plt.close()
                        df_round = pd.concat([df_round, df_client])

                mean_cs_global = sum(list_confidence_score)/len(list_confidence_score)
                ax = sns.displot(df_round, x='Cs', kind="kde")
                for ax in ax.axes.flat:
                            # print (ax.lines)
                    for line in ax.lines:
                        x = line.get_xdata() # Get the x data of the distribution
                        y = line.get_ydata() # Get the y data of the distribution
                maxid = np.argmax(y) 
                peak_global = y[maxid]
                plt.close()
                predicted_normal = []
                predicted_attacker = []
                list_peak_normal = []
                list_peak_attacker = []
                list_cs_normal = []
                list_cs_attacker = []
                # normal_models = []
                # list_acc_attacker = []
                for idx, client in enumerate(self.selected_clients):
                    # if self.option['agg_algorithm'] == "peak_or_cs_choose_normal":
                    #     if list_confidence_score[idx] > mean_cs_global or list_peak[idx] < peak_global:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "peak_or_cs_remove_attacker": 
                    #     if list_confidence_score[idx] < mean_cs_global or list_peak[idx] > peak_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "peak_and_cs_choose_normal": 
                    #     if list_confidence_score[idx] > mean_cs_global and list_peak[idx] < peak_global:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "peak_and_cs_remove_attacker": 
                    #     if list_confidence_score[idx] < mean_cs_global and list_peak[idx] > peak_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "peak_compare_avg_peak": 
                    #     if list_peak[idx] > peak_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "cs_compare_avg_cs":
                    #     if list_confidence_score[idx] < mean_cs_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "remove_all_attacker_agg_all_clean":
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "aggregate_attacker_by_noise_rate_(1-pi)":
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_csi":
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] in ["aggregate_attacker_exactly_by_(csi+0.05)",
                    #                                       "aggregate_attacker_exactly_by_csi",
                    #                                       "aggregate_attacker_exactly_by_(csi/max_cs)",
                    #                                       "aggregate_attacker_exactly_by_(acci+0.05)",
                    #                                       "aggregate_attacker_exactly_by_(acci/(maxacc+0.05))",
                    #                                       "aggregate_attacker_exactly_by_(acc_before_trains[id] - 0.1)",
                    #                                       "aggregate_attacker_exactly_by_((csi+0.05)^2)",
                    #                                       "aggregate_attacker_exactly_by_((csi+0.05)^(1/2))",
                    #                                       "aggregate_attacker_exactly_by_((csi+0.05)^3)",
                    #                                       "aggregate_attacker_exactly_by_((csi+0.05)^(1/3))",
                    #                                       "aggregate_attacker_exactly_by_(tanh3(csi+0.05))",
                    #                                       "aggregate_attacker_exactly_by_(tanhe(csi+0.05))",
                    #                                       "aggregate_attacker_exactly_by_(tanhe(csi+0.05*mincs/maxcs))"
                    #                                       ]:
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                   
                    # if self.option['agg_algorithm'] in ["peak_and_cs_choose_attacker_aggregate_attacker_by_cs_csi",
                                                        #   "peak_and_cs_choose_attacker_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))",
                    if self.option['agg_algorithm'] == "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)":
                        if list_confidence_score[idx] < mean_cs_global and list_peak[idx] > peak_global:
                            predicted_attacker.append(client)
                            list_peak_attacker.append(list_peak[idx])
                            list_cs_attacker.append(list_confidence_score[idx])
                        else:
                            predicted_normal.append(client)
                            list_peak_normal.append(list_peak[idx])
                            list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] in ["peak_or_cs_choose_normal_aggregate_attacker_by_cs_csi",
                    #                                       "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05)",
                    #                                       "peak_or_cs_choose_normal_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))",
                    #                                       "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05*mincs/maxcs)"] :
                    #     if list_confidence_score[idx] > mean_cs_global or list_peak[idx] < peak_global:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "peak_choose_attacker_aggregate_attacker_by_cs_csi":
                    #     if list_peak[idx] > peak_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "cs_choose_attacker_aggregate_attacker_by_cs_csi":
                    #     if list_confidence_score[idx] < mean_cs_global:
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "aggregate_attacker_by_noise_rate_(1-pi+)":
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                    # elif self.option['agg_algorithm'] == "remove_attacker_have_classes_of_clean":
                    #     client_classes = [[5, 6, 9], [0, 2, 6], [2, 3, 9], [2, 5, 6], [2, 7],
                    #                       [1, 3, 8], [0, 2, 7], [0, 5, 7], [6, 7, 8], [1, 3, 4],
                    #                       [7, 9], [0, 4, 5], [2, 8], [2, 8], [6, 8], [0, 1, 3],
                    #                       [2, 4, 6], [6, 8, 9], [1, 4, 5], [6, 7, 9], [8, 9],
                    #                       [2, 4, 7], [2, 3, 7], [1, 2], [0, 4, 6], [1, 3, 9],
                    #                       [5, 9], [5, 8, 9], [5, 6], [6, 9], [3, 7, 8],
                    #                       [2, 5, 6], [7, 8, 9], [2, 3, 5], [0, 7, 9],
                    #                       [0, 4, 6], [4], [1, 9], [1, 5], [2, 6, 7],
                    #                       [2, 5], [0, 1, 6], [1, 3, 5], [2, 4, 8], [0, 1, 2],
                    #                       [0, 2, 8], [4, 9], [0, 6, 7], [5, 6, 7], [0, 3, 6]]
                    #     # client_classes = [[1, 5, 8], [1, 2], [0, 6], [1, 2, 7], [2, 4],
                    #     #                   [0, 3], [3, 5, 6], [5, 8, 9], [0, 1, 7], [0, 1, 3],
                    #     #                   [2, 4, 7], [3, 6], [0, 2, 4], [0, 3, 6], [5, 7, 9],
                    #     #                   [3, 5, 9], [7, 9], [0, 1, 2], [3, 8], [0, 2, 6],
                    #     #                   [5, 7, 8], [5, 8, 9], [1, 2], [1, 6, 7], [1, 4, 6],
                    #     #                   [3, 6, 8], [1, 2, 9], [1, 2, 8], [1, 4, 5], [1, 8]]
                    #     if self.clients[client].train_data.client_type == "attacker":
                    #         predicted_attacker.append(client)
                    #         list_peak_attacker.append(list_peak[idx])
                    #         list_cs_attacker.append(list_confidence_score[idx])
                    #     else:
                    #         predicted_normal.append(client)
                    #         list_peak_normal.append(list_peak[idx])
                    #         list_cs_normal.append(list_confidence_score[idx])
                
                
                # if self.option['agg_algorithm'] == "remove_attacker_have_classes_of_clean":
                #     print(f"Predicted normal: {predicted_normal}")
                #     print(f"Predicted attacker: {predicted_attacker}")
                #     normal_classes = []
                #     for normal in predicted_normal:
                #         for class_i in client_classes[int(normal)]:
                #             if class_i not in normal_classes:
                #                 normal_classes.append(class_i)
                #     for attacker in predicted_attacker:
                #         for class_i in client_classes[int(attacker)]:
                #             if class_i not in normal_classes:
                #                 predicted_normal.append(attacker)
                #                 predicted_attacker.remove(attacker)
                #                 break
                #     print(f"Predicted normal: {predicted_normal}")
                #     print(f"Predicted attacker: {predicted_attacker}")
                # sum_sample = sum([self.client_vols[cid] for cid in predicted_normal])
                # p_ = []
                # for client in self.selected_clients:
                #     if client in predicted_normal:
                #         p_.append(1.0 * self.client_vols[client]/sum_sample)
                #     else:
                #         p_.append(0)
                # if self.option['agg_algorithm'] == "aggregate_attacker_by_noise_rate_(1-pi)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     for client in self.selected_clients:
                #         p_.append((1-self.clients[client].train_data.dirty_rate)*self.client_vols[client]/sum_sample)
                # if self.option['agg_algorithm'] == "aggregate_attacker_by_noise_rate_(1-pi+)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     for client in self.selected_clients:
                #         p_.append(1-self.clients[client].train_data.dirty_rate + self.client_vols[client]/sum_sample)
                # if self.option['agg_algorithm'] in ["peak_and_cs_choose_attacker_aggregate_attacker_by_cs_csi", 
                                                    # "peak_or_cs_choose_normal_aggregate_attacker_by_cs_csi",
                                                    # "peak_choose_attacker_aggregate_attacker_by_cs_csi",
                                                    # "cs_choose_attacker_aggregate_attacker_by_cs_csi",
                                                    # "aggregate_attacker_exactly_by_csi",
                                                    # "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05)",
                                                    # "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05*mincs/maxcs)",
                                                    # "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)"]:
                if self.option['agg_algorithm'] == "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)":
                    p_ = []
                    sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                    id_attacker = 0
                    for client in self.selected_clients:
                        if client in predicted_normal:
                            p_.append(self.client_vols[client]/sum_sample)
                        else:
                            max_cs = max(list_confidence_score)
                            min_cs = min(list_confidence_score)
                            csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                            # if self.option['agg_algorithm'] == "peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05)":
                            #     p_.append((csi+0.05) * self.client_vols[client]/sum_sample)
                            # elif self.option['agg_algorithm'] in ["peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05*mincs/maxcs)",
                            #                                       "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)"]:
                            #     print(self.option['ours_beta'])
                            p_.append((csi+self.option['ours_beta']*min_cs/max_cs) * self.client_vols[client]/sum_sample)
                            # else:
                            #     p_.append(csi * self.client_vols[client]/sum_sample)

                            id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(csi+0.05)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append((csi+0.05) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_((csi+0.05)^2)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append(((csi+0.05)**2) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_((csi+0.05)^(1/2))":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append(((csi+0.05)**(1/2.0)) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1 
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_((csi+0.05)^3)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append(((csi+0.05)**(3)) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_((csi+0.05)^(1/3))":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append(((csi+0.05)**(1/3.0)) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(tanh3(csi+0.05))":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append((np.tanh(3*(csi+0.05))) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(tanhe(csi+0.05))":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append((np.tanh(np.exp(1)*(csi+0.05)) * self.client_vols[client]/sum_sample))
                #             id_attacker +=1
                # if self.option['agg_algorithm'] in ["aggregate_attacker_exactly_by_(tanhe(csi+0.05*mincs/maxcs))",
                #                                     "peak_or_cs_choose_normal_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))",
                #                                     "peak_and_cs_choose_attacker_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))"]:
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                #             p_.append((np.tanh(np.exp(1)*(csi+0.05*min_cs/max_cs)) * self.client_vols[client]/sum_sample))
                #             id_attacker +=1
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(csi/max_cs)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     id_attacker = 0
                #     for client in self.selected_clients:
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(list_confidence_score)
                #             min_cs = min(list_confidence_score)
                #             csi = list_cs_attacker[id_attacker]/max_cs
                #             p_.append((csi) * self.client_vols[client]/sum_sample)
                #             id_attacker +=1

                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(acci+0.05)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     for id,client in enumerate(self.selected_clients):
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_cs = max(acc_before_trains)
                #             min_cs = min(acc_before_trains)
                #             csi = (acc_before_trains[id] - min_cs)/(max_cs-min_cs)
                #             p_.append((csi+0.05) * self.client_vols[client]/sum_sample)
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(acc_before_trains[id] - 0.1)":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     for id,client in enumerate(self.selected_clients):
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             # max_cs = max(acc_before_trains)
                #             # min_cs = min(acc_before_trains)
                #             # csi = (acc_before_trains[id] - min_cs)/(max_cs-min_cs)
                #             p_.append(abs(acc_before_trains[id] - 0.1) * self.client_vols[client]/sum_sample)
                # if self.option['agg_algorithm'] == "aggregate_attacker_exactly_by_(acci/(maxacc+0.05))":
                #     p_ = []
                #     sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                #     for id,client in enumerate(self.selected_clients):
                #         if client in predicted_normal:
                #             p_.append(self.client_vols[client]/sum_sample)
                #         else:
                #             max_acc = max(acc_before_trains)
                #             # min_acc = min(list_acc_attacker)
                #             acci = acc_before_trains[id]/(max_acc + 0.05)
                #             p_.append(acci * self.client_vols[client]/sum_sample)
                # all_client = {}
                # for client in self.selected_clients:
                #     all_client[client] = self.clients[client].train_data.dirty_rate
                # print("client_rate", all_client)
                # print("acc_client", acc_before_trains)
                    
                # print('p_ = ', p_)
                if len(predicted_normal) > 0:
                    # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/sum_sample for cid in predicted_normal])
                    self.model = self.aggregate(models, p = p_)
                ours_server_time = time.time() - ours_server_time_start
                self.computation_time[round]['server_aggregation_time'] = ours_server_time
                # path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/type_noise_{}/proportion_{}/num_malicious_{}/csv/{}'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0],self.option['outside_noise'], self.option['proportion']*50, self.option['num_malicious'], 0)
                # file = f"epoch{round}.csv"
                # path_csv = os.path.join(path_, file)
                # df = pd.read_csv(path_csv, index_col=0)
                
                # Y = df.loc[:,["target"]].to_numpy()
                attacker_idx = []
                for idx, client in enumerate(self.selected_clients):
                    # if Y[y] == "attacker":
                    if self.clients[client].train_data.client_type == "attacker":
                        attacker_idx.append(client)
                # attacker_idx = np.nonzero(Y_)[0]        
                print(f"Attacker idx: {attacker_idx}")
                print(f"Predicted attacker: {predicted_attacker}")
                    
                
                true_pred = list(set(attacker_idx) & set(predicted_attacker))
                print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                wrong_pred_attacker = []
                for client in predicted_attacker:
                    if client not in attacker_idx:
                        wrong_pred_attacker.append(client)
                print("Wrong prediction attackers: {}/{}".format(len(wrong_pred_attacker),len(predicted_attacker)))
                wrong_pred_normal = []
                for client in predicted_normal:
                    if client in attacker_idx:
                        wrong_pred_normal.append(client)
                print("Wrong prediction normals: {}/{}".format(len(wrong_pred_normal),len(predicted_normal)))
                print('\n')
                
                dictionary = {
                "Real attacker": [int(i) for i in attacker_idx],
                "Global peak": peak_global,
                "List peak predicted normal": list_peak_normal,
                "List peak predicted attacker": list_peak_attacker,
                "Global cs": mean_cs_global,
                "List cs predicted normal": list_cs_normal,
                "List cs predicted attacker": list_cs_attacker,
                "Predicted normal": [int(i) for i in predicted_normal],
                "Predicted attacker": [int(i) for i in predicted_attacker],
                "true prediction attacker": [int(i) for i in true_pred],
                "wrong prediction attacker": [int(i) for i in wrong_pred_attacker],
                "wrong prediction normal": [int(i) for i in wrong_pred_normal],
                }
                path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                # listObj = []
                # if round != 1:
                #     with open(path_ + 'log.json') as fp:
                #         listObj = json.load(fp)
                
                # listObj.append(dictionary)
                
                # with open(path_ + 'log.json', 'w') as json_file:
                #     json.dump(listObj, json_file, indent=4)
                if self.option['log_time'] == 1:
                    with open(path_ + 'log_time.json', 'w') as f:
                        json.dump(self.computation_time, f, indent=4)
                # thresholds
                # 
                # cluster_list = [[0,5],   #  0/2
                #                 [1,6,7,8,9,10,11], #6,8   2/7
                #                 [2,12,13,14,15,16,17,18,19,20], #12,13,15,17,19   5/10
                #                 [3,21,22,23,24,25,26,27,28,29,30], #3,25,26,30    4/11
                #                 [4,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]] #4,32,37,39,41,45,46,47,48    9/20
                # # attacker= [3, 4, 6,8,12,13,15,17,19,25,26,30,32,37,39,41,45,46,47,48]
                # cluster_selected_client_list = [[],[],[],[],[]]
                # for idx,client in enumerate(self.selected_clients):
                #     for i in range(5):
                #         if client in cluster_list[i]:
                #             cluster_selected_client_list[i].append(idx)
                # chosen_cluster_round = []
                # dict_log_round = {}
                # min_avg_cs_clean = 1.0
                # uncertainty_cluster_client = []
                # # for iter in range(2):
                # #     if iter == 0:
                # #         min_ = 1.0
                # #         for i in range(5):
                # #             if len(cluster_selected_client_list[i]) == 1:
                # #                 uncertainty_cluster_client.append(i)
                # #                 continue
                # #             chosen_cluster_i, dict_log_i, min_avg_cs_clean_cluster_i = self.cluster_2(cluster_selected_client_list[i], models, acc_before_trains, round, iter, i)
                # #             if min_avg_cs_clean_cluster_i == 1.0:
                # #                 uncertainty_cluster_client.append(i)
                # #                 continue
                # #             min_ = min(min_, min_avg_cs_clean_cluster_i)
                # #             for client in chosen_cluster_i:
                # #                 if client not in chosen_cluster_round:
                # #                     chosen_cluster_round.append(client)
                # #             dict_log_round[f"cluster class {i}"] = dict_log_i
                # #     else:
                # #         if min_ != 1.0: 
                # #             self.min_avg_cs_clean_threshold = self.min_avg_cs_clean_threshold*0.4 + min_*0.6
                # #         for i in range(5):
                # #             if i in uncertainty_cluster_client:
                # #                 chosen_cluster_i, dict_log_i, min_avg_cs_clean_cluster_i = self.cluster_2(cluster_selected_client_list[i], models, acc_before_trains, round, iter, i)
                # #                 for client in chosen_cluster_i:
                # #                     if client not in chosen_cluster_round:
                # #                         chosen_cluster_round.append(client)
                # #                 dict_log_round[f"cluster class {i}"] = dict_log_i
                # # cluster_client_update = []
                # # for i in range(5):
                # #     if self.frequent_update_threshold[i] <= 1:
                # #         cluster_client_update.append(self.min_avg_cs_clean_threshold[i]) 
                # for i in range(5):
                #     if self.frequent_update_threshold[i] == 3:
                #         # self.min_avg_cs_clean_threshold[i] = (sum(self.min_avg_cs_clean_threshold) - self.min_avg_cs_clean_threshold[i])/4
                #         self.min_avg_cs_clean_threshold[i] = (sum(self.cs_clean_recently[i])/len(self.cs_clean_recently[i]))*0.6 + self.min_avg_cs_clean_threshold[i]*0.4
                #         self.frequent_update_threshold[i] = 0
                # for i in range(5):
                #     chosen_cluster_i, dict_log_i, min_avg_cs_clean_cluster_i = self.cluster_2(cluster_selected_client_list[i], models, acc_before_trains, round, iter, i)
                #     if min_avg_cs_clean_cluster_i == 2:
                #         self.frequent_update_threshold[i] += 1
                #     else:
                #         self.frequent_update_threshold[i] = 0
                #         self.min_avg_cs_clean_threshold[i] = self.min_avg_cs_clean_threshold[i]*0.4 + min_avg_cs_clean_cluster_i*0.6
                #     for client in chosen_cluster_i:
                #         if client not in chosen_cluster_round:
                #             chosen_cluster_round.append(client)
                #     dict_log_round[f"cluster class {i}"] = dict_log_i 
                
                # print('self.min_avg_cs_clean_threshold',self.min_avg_cs_clean_threshold)
                # print('self.frequent_update_threshold',self.frequent_update_threshold)
                # print('self.cs_clean_recently',self.cs_clean_recently)
                
                # # models_chosen_round = [models[i] for i in chosen_cluster_round]
                # # self.model = self.agg_fuction(models_chosen_round)
                # cluster_client_ = []
                # for idx, cid in enumerate(self.selected_clients):
                #     if idx in chosen_cluster_round:
                #         zmx,czxm,czxmcxncmzxnczxm,ncmzxcnzxmcxmz,cnzxm,xzncmzx,cnzxm,cnzxmcnxz.append(cid)
                # sum_sample = sum([self.client_vols[cid] for cid in cluster_client_])
                # if len(cluster_client_) > 0:
                #     self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/sum_sample for cid in cluster_client_])
                # path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'])
                # listObj = []
                # if round != 1:
                #     with open(path_ + 'log.json') as fp:
                #         listObj = json.load(fp)
                
                # listObj.append(dict_log_round)
                
                # with open(path_ + 'log.json', 'w') as json_file:
                #     json.dump(listObj, json_file, indent=4)
                # # models_cluster_0 = [models[i] for i in cluster_0]
                # # models_cluster_1 = [models[i] for i in cluster_1]
                # # # models_cluster_2 = [models[i] for i in cluster_2]
                
                # # Avg_confidence_score_list = []
                # # for idx,client in enumerate(self.selected_clients):
                # #     cs_client = sum(self.confidence_score[round][int(client)].values())/len(self.confidence_score[round][int(client)])
                # #     Avg_confidence_score_list.append(cs_client)
          
                # # # acc_before_train_cluster_0 = [acc_before_trains[i] for i in cluster_0]
                # # # acc_before_train_cluster_1 = [acc_before_trains[i] for i in cluster_1]
                # # acc_before_train_cluster_0 = [Avg_confidence_score_list[i] for i in cluster_0]
                # # acc_before_train_cluster_1 = [Avg_confidence_score_list[i] for i in cluster_1]
                
                # # Avg_acc_before_train_cluster_0 = sum(acc_before_train_cluster_0)/len(acc_before_train_cluster_0)
                # # Avg_acc_before_train_cluster_1 = sum(acc_before_train_cluster_1)/len(acc_before_train_cluster_1)
                
                # # # aggregate_model_cluster_0 = self.agg_fuction(models_cluster_0)
                # # # aggregate_model_cluster_1 = self.agg_fuction(models_cluster_1)                
                # # # aggregate_model_cluster_2 = self.agg_fuction(models_cluster_2)
                
                # # print(f'Avg_acc_before_train_cluster_0 : {Avg_acc_before_train_cluster_0}')
                # # print(f'Avg_acc_before_train_cluster_1 : {Avg_acc_before_train_cluster_1}')
                # # if Avg_acc_before_train_cluster_0 > Avg_acc_before_train_cluster_1: 
                # #     if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 > 0.05:
                # #         # self.model = aggregate_model_cluster_0
                # #         cluster_client_ = []
                # #         for idx, cid in enumerate(self.selected_clients):
                # #             if idx in cluster_0:
                # #                 cluster_client_.append(cid)
                # #         sum_sample = sum([self.client_vols[cid] for cid in cluster_client_])
                # #         self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/sum_sample for cid in cluster_client_])
                # #         chosen_cluster = cluster_0
                # #         attacker_cluster = cluster_1
                # #     else: 
                # #         # self.model = self.agg_fuction(models)
                # #         self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
                        
                # #         chosen_cluster = [y for y in range(len(y_pred))]
                # #         attacker_cluster = []
                # # else:
                # #     if Avg_acc_before_train_cluster_1 - Avg_acc_before_train_cluster_0 > 0.05:
                # #         # self.model = aggregate_model_cluster_1
                # #         cluster_client_ = []
                # #         for idx, cid in enumerate(self.selected_clients):
                # #             if idx in cluster_1:
                # #                 cluster_client_.append(cid)
                # #         sum_sample = sum([self.client_vols[cid] for cid in cluster_client_])
                # #         self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/sum_sample for cid in cluster_client_])
                # #         chosen_cluster = cluster_1
                # #         attacker_cluster = cluster_0
                # #     else:
                # #         # self.model = self.agg_fuction(models)
                # #         self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
                        
                # #         chosen_cluster = [y for y in range(len(y_pred))]
                # #         attacker_cluster = []
                    
                # # # test_metric_all, test_loss_all, inference_time_all = self.test(self.agg_fuction(models), device=torch.device('cuda'))
                # # # test_metric_0, test_loss_0, inference_time_0 = self.test(aggregate_model_cluster_0, device=torch.device('cuda'))
                # # # test_metric_1, test_loss_1, inference_time_1 = self.test(aggregate_model_cluster_1, device=torch.device('cuda'))
                # # # test_metric_2, test_loss_2, inference_time_2 = self.test(aggregate_model_cluster_2, device=torch.device('cuda'))
                # # # print('\n')
                # # # print(f"Test acc of all: {test_metric_all}")
                # # # print(f"Test acc of cluster 0: {test_metric_0}")
                # # # print(f"Test acc of cluster 1: {test_metric_1}")
                # # # print(f"Test acc of cluster 2: {test_metric_2}")
                
                # # # cluster_list = [cluster_0, cluster_1, cluster_2]
                # # # acc_list = [test_metric_0, test_metric_1, test_metric_2]
                # # # agg_model_list = [aggregate_model_cluster_0, aggregate_model_cluster_1, aggregate_model_cluster_2]
                # # # max_idx = acc_list.index(max(acc_list))
                # # # min_idx = acc_list.index(min(acc_list))
                # # # for mid_idx in range(len(acc_list)):
                # # #     if mid_idx != max_idx and mid_idx != min_idx:
                # # #         break
                # # # if acc_list[max_idx] - acc_list[mid_idx] <= 0.05:
                # # #     self.model = self.aggregate([agg_model_list[max_idx], agg_model_list[mid_idx]], p = [0.9, 0.1])
                # # # else: 
                # # #     self.model = agg_model_list[max_idx]
                    
                # # test_metric_agg, test_loss_agg, inference_time_agg = self.test(self.model, device=torch.device('cuda'))
                # # print('\n')
                # # # print(f"Choose cluster {max_idx} and {mid_idx} to aggregate")
                # # print(f"Test acc of agg model: {test_metric_agg}")
                # # # true_pred = list(set(attacker_idx) & set(cluster_list[min_idx]))
                # # true_pred = list(set(attacker_idx) & set(attacker_cluster))
                # # print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                # # # if test_metric_1 >= test_metric_2:
                # # #     self.model = aggregate_model_cluster_1
                # # #     print('\n')
                # # #     print("Choose cluster 1 to aggregate")
                # # #     true_pred = list(set(attacker_idx) & set(cluster_2))
                # # #     print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                # # # else: 
                # # #     self.model = aggregate_model_cluster_2
                # # #     print('\n')
                # # #     print("Choose cluster 2 to aggregate")
                # # #     true_pred = list(set(attacker_idx) & set(cluster_1))
                # # #     print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                # # dictionary = {
                # # "round": round,
                # # "attacker_idx": attacker_idx.tolist(),
                # # "cluster 0": cluster_0,
                # # "cluster 1": cluster_1,
                # # #   "cluster 2": cluster_2,
                # # #   "test acc all": test_metric_all,
                # # #   "test acc cluster 0": test_metric_0,
                # # #   "test acc cluster 1": test_metric_1,
                # # #   "test acc cluster 2": test_metric_2,
                # # "Avg_acc_before_train_cluster_0": Avg_acc_before_train_cluster_0,
                # # "Avg_acc_before_train_cluster_1": Avg_acc_before_train_cluster_1,
                # # "Chosen cluster": chosen_cluster,
                # # "attacker_cluster": attacker_cluster,
                # # "test acc after agg": test_metric_agg,
                # # "true prediction attacker": [int(i) for i in true_pred],
                # # }
                # # path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'])
                # # listObj = []
                # # if round != 1:
                # #     with open(path_ + 'log.json') as fp:
                # #         listObj = json.load(fp)
                
                # # listObj.append(dictionary)
                
                # # with open(path_ + 'log.json', 'w') as json_file:
                # #     json.dump(listObj, json_file, indent=4)
            
            else:
                # self.model = self.agg_fuction(models)
                avg_confidence_score_list = []
                for idx,client in enumerate(self.selected_clients):
                    # if idx in list_idx:
                    confidence_score_dict_client = self.confidence_score[round][int(client)]
                    avg_confidence_score_list.append(sum(confidence_score_dict_client.values())/len(confidence_score_dict_client))
                all_client = {}
                for id,client in enumerate(self.selected_clients):
                    all_client[int(client)] = [float(self.clients[client].train_data.dirty_rate), float(loss_before_trains[id]), float(acc_before_trains[id]), float(avg_confidence_score_list[id])]
                print("client_rate", all_client)
                path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                # listObj = []
                # if round != 1:
                #     with open(path_ + 'log.json') as fp:
                #         listObj = json.load(fp)
                
                # listObj.append(all_client)
                
                # with open(path_ + 'log.json', 'w') as json_file:
                #     json.dump(listObj, json_file, indent=4)
                # print("acc_client", acc_before_trains)
                # print("loss_client", loss_before_trains)
                fedavg_time_start = time.time()
                self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
                fedavg_time = time.time() - fedavg_time_start
                self.computation_time[round]['server_aggregation_time'] = fedavg_time
                
                if self.option['log_time'] == 1:
                    with open(path_ + 'log_time.json', 'w') as f:
                        json.dump(self.computation_time, f, indent=4)
            
            print(f'Done aggregate at round {self.current_round}')
        else:
            other_defense_time_start = time.time()
            self.model = self.agg_fuction(models)
            other_defense_time = time.time() - other_defense_time_start
            self.computation_time[round]['server_aggregation_time'] = other_defense_time
            path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
            if self.option['log_time'] == 1:
                with open(path_ + 'log_time.json', 'w') as f:
                    json.dump(self.computation_time, f, indent=4)
                
            print(f'Done aggregate at round {self.current_round}')
            
            
        
        # attacked_class_accuracies = []
        # actuals, predictions = self.test_label_predictions(copy.deepcopy(self.model), device0)
        # classes = list(i for i in range(10))
        # print('{0:10s} - {1}'.format('Class','Accuracy'))
        # # for i, r in enumerate(confusion_matrix(actuals, predictions)):
        # #     print('{} - {:.2f}'.format(classes[i], r[i]/np.sum(r)*100))
        # #     if i in self.option['attacked_class']:
        # #         attacked_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))

        
        # path_csv = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv'.format(len(self.option['attacked_class']),self.option['dirty_rate'][0],self.option['proportion']*50,self.option['num_malicious'])
        
        # with open(path_csv + '/' + 'confusion_matrix.csv', 'a', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     pd.DataFrame(data= confusion_matrix(actuals, predictions),
        #                  index=['Class' + str(i) for i in range(10)],
        #                  columns= ['Class' + str(i) for i in range(10)]).to_csv(path_csv + '/' + 'round_{}.csv'.format(round))
            
        #     for i, r in enumerate(confusion_matrix(actuals, predictions)):
        #         data = []
        #         print('{} - {:.2f}'.format(classes[i], r[i]/np.sum(r)*100))
        #         data.append(classes[i])
        #         data.append(r[i]/np.sum(r)*100)
        #         if i in self.option['attacked_class']:
        #             attacked_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
        #             data.append('attacked')
        #         else:
        #             data.append('clean')
        #         writer.writerow(data)
        #     writer.writerow(['','',''])
        #     writer.writerow(['','',''])
            
        
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
            eval_value, loss, confidence_score_dict = c.test(self.model, dataflag, device=device)
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
            # peer_grad = []
            # # confidence_score_dict = {}
            # # for idx in self.train_data.idxs:
            # #     confidence_score_dict[idx] = 0
            for iter in range(self.epochs):
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    # optimizer.step()
                    # loss, confidence_score_list, idx_list = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    loss = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    loss.backward()
                    # for cs in confidence_score_list:
                    #     if cs >= 1.0:
                    #         print(cs)
                    #         print("> 1")
                    # if len(self.train_data) < 64:
                    #     print('confidence_score_list', confidence_score_list)
                    #     print('idx_list', idx_list)
                    
                    # Get gradient
                    # for i, (name, params) in enumerate(model.named_parameters()):
                    #     if params.requires_grad:
                    #         if iter == 0 and batch_id == 0:
                    #             peer_grad.append(params.grad.clone())
                    #         else:
                    #             peer_grad[i]+= params.grad.clone()
                    optimizer.step()
                    # for i,idx in enumerate(idx_list):
                    #     confidence_score_dict[idx]+=confidence_score_list[i]
                        # confidence_score_dict[idx].append(confidence_score_list[i]) 
                    # print('self.confidence_score_dict', self.confidence_score_dict)
            # print('self.confidence_score_dict', self.confidence_score_dict)
            
            # return np.array(0), self.train_data.idxs
            # train function at clients
            # if self.option['percent_noise_remove'] != 0:
            #     self.train_data.reverse_idx()
            # return peer_grad#, confidence_score_dict
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
        confidence_score_dict = {}
        dataset = self.train_data
        model = model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss, _,confidence_score_list, idx_list  = self.calculator.test_client(model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
            for i,idx in enumerate(idx_list):
                confidence_score_dict[idx] = confidence_score_list[i]
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss, confidence_score_dict


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
        calculate_cs_time_start = time.time()
        acc_before_train, loss_before_train,confidence_score_dict = self.test(model, device)
        calculate_cs_time = time.time() - calculate_cs_time_start
        
        train_time_start = time.time()
        self.train(model, device, round)
        train_time = time.time() - train_time_start
        # peer_grad = self.train(model, device, round)
        # acc_before_train, loss_before_train,confidence_score_dict = self.test(model, device)
        # cpkg = self.pack(model, peer_grad, acc_before_train, loss_before_train, confidence_score_dict)
        cpkg = self.pack(model, acc_before_train, loss_before_train, confidence_score_dict, calculate_cs_time, train_time)
        
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
