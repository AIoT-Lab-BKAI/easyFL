from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
from utils import fmodule
import torch
import wandb
import time
import json
from utils.plot_pca import *
from sklearn.cluster import KMeans


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['alpha']
        self.alpha = option['alpha']
        # self.h  = self.model.zeros_like()
        # self.h = self.model.state_dict().copy()
        self.h = []
        for param in self.model.parameters():
            self.h.append(param.data)
        
    # def iterate(self, t):
    #     self.selected_clients = self.sample()
    #     models, train_losses = self.communicate(self.selected_clients)
    #     if not self.selected_clients: return
    #     start = time.time()
    #     self.model = self.aggregate(models)
    #     end = time.time()
    #     if self.wandb:
    #         wandb.log({"Aggregation_time": end-start})
    #     return

    def aggregate(self, models):
        # self.h = self.h - self.alpha * 1.0 / self.num_clients * (fmodule._model_sum(models) - self.model)
        # new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        # return new_model
        num_participants = len(models)
        sum_theta = []
        for idx, param in enumerate(models[0].parameters()):
                sum_theta.append(param.data)
        for client_theta in models[1:]:
            for idx, param in enumerate(client_theta.parameters()):
                sum_theta[idx] += param.data

        delta_theta = []
        for idx, param in enumerate(self.model.parameters()):
            delta_theta.append(sum_theta[idx] - param.data)

        for idx, param in enumerate(self.model.parameters()):
            self.h[idx] -= self.alpha * (1.0/self.num_clients) * delta_theta[idx]
        
        
        for idx, param in enumerate(self.model.parameters()):
            param.data = (1./num_participants) * sum_theta[idx] - (1./self.alpha) *  self.h[idx]

        return self.model
        
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
        models, peer_grads, acc_before_trains = self.communicate(self.selected_clients, pool)
        peers_types = [self.clients[id].train_data.client_type for id in self.selected_clients]
        plot_updates_components(copy.deepcopy(self.model), peer_grads, peers_types, epoch=round, proportion = self.option['proportion'], attacked_class = self.option['attacked_class'],dirty_rate=self.option['dirty_rate'][0],num_malicious=self.option['num_malicious'], agg_algorithm=self.option['agg_algorithm'], algorithm= self.option['algorithm'])
        
        if self.option['agg_algorithm'] == "cluster_2_0.05":
            path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv/{}'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'], 0)
            
            file = f"epoch{round}.csv"
            path_csv = os.path.join(path_, file)
            df = pd.read_csv(path_csv, index_col=0)
            X = df.loc[:, ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"]].to_numpy()
            Y = df["target"].to_numpy()
            Y_ = []
            for y in Y:
                if y == "attacker":
                    Y_.append(1)
                else:
                    Y_.append(0)
            attacker_idx = np.nonzero(Y_)[0]        
            print(f"Attacker idx: {attacker_idx}")
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
            y_pred = kmeans.labels_
            cluster_0 = []
            cluster_1 = []
            # cluster_2 = []
            for y in range(len(y_pred)):
                if y_pred[y] == 0:
                    cluster_0.append(y)
                elif y_pred[y] == 1:
                    cluster_1.append(y)
                # else: 
                #     cluster_2.append(y)
            print('\n')
            print(f"Cluster 0: {cluster_0}")
            print(f"Cluster 1: {cluster_1}")
        # print(f"Cluster 2: {cluster_2}")
         
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
            # self.model = self.agg_fuction(models)
            if self.option['agg_algorithm'] == "cluster_2_0.05":   
                models_cluster_0 = [models[i] for i in cluster_0]
                models_cluster_1 = [models[i] for i in cluster_1]
                # models_cluster_2 = [models[i] for i in cluster_2]
                
                acc_before_train_cluster_0 = [acc_before_trains[i] for i in cluster_0]
                acc_before_train_cluster_1 = [acc_before_trains[i] for i in cluster_1]
                
                Avg_acc_before_train_cluster_0 = sum(acc_before_train_cluster_0)/len(acc_before_train_cluster_0)
                Avg_acc_before_train_cluster_1 = sum(acc_before_train_cluster_1)/len(acc_before_train_cluster_1)
                
                # aggregate_model_cluster_0 = self.aggregate(models_cluster_0)
                # aggregate_model_cluster_1 = self.aggregate(models_cluster_1)
                # aggregate_model_cluster_2 = self.aggregate(models_cluster_2)
                
                print(f'Avg_acc_before_train_cluster_0 : {Avg_acc_before_train_cluster_0}')
                print(f'Avg_acc_before_train_cluster_1 : {Avg_acc_before_train_cluster_1}')
                if Avg_acc_before_train_cluster_0 > Avg_acc_before_train_cluster_1: 
                    if Avg_acc_before_train_cluster_0 - Avg_acc_before_train_cluster_1 > 0.05:
                        self.model = self.aggregate(models_cluster_0)
                        chosen_cluster = cluster_0
                        attacker_cluster = cluster_1
                    else: 
                        self.model = self.aggregate(models)
                        chosen_cluster = [y for y in range(len(y_pred))]
                        attacker_cluster = []
                else:
                    if Avg_acc_before_train_cluster_1 - Avg_acc_before_train_cluster_0 > 0.05:
                        self.model = self.aggregate(models_cluster_1)
                        chosen_cluster = cluster_1
                        attacker_cluster = cluster_0
                    else:
                        self.model = self.aggregate(models)
                        chosen_cluster = [y for y in range(len(y_pred))]
                        attacker_cluster = []
                    
                # test_metric_all, test_loss_all, inference_time_all = self.test(self.agg_fuction(models), device=torch.device('cuda'))
                # test_metric_0, test_loss_0, inference_time_0 = self.test(aggregate_model_cluster_0, device=torch.device('cuda'))
                # test_metric_1, test_loss_1, inference_time_1 = self.test(aggregate_model_cluster_1, device=torch.device('cuda'))
                # test_metric_2, test_loss_2, inference_time_2 = self.test(aggregate_model_cluster_2, device=torch.device('cuda'))
                # print('\n')
                # print(f"Test acc of all: {test_metric_all}")
                # print(f"Test acc of cluster 0: {test_metric_0}")
                # print(f"Test acc of cluster 1: {test_metric_1}")
                # print(f"Test acc of cluster 2: {test_metric_2}")
                
                # cluster_list = [cluster_0, cluster_1, cluster_2]
                # acc_list = [test_metric_0, test_metric_1, test_metric_2]
                # agg_model_list = [aggregate_model_cluster_0, aggregate_model_cluster_1, aggregate_model_cluster_2]
                # max_idx = acc_list.index(max(acc_list))
                # min_idx = acc_list.index(min(acc_list))
                # for mid_idx in range(len(acc_list)):
                #     if mid_idx != max_idx and mid_idx != min_idx:
                #         break
                # if acc_list[max_idx] - acc_list[mid_idx] <= 0.05:
                #     self.model = self.aggregate([agg_model_list[max_idx], agg_model_list[mid_idx]], p = [0.9, 0.1])
                # else: 
                #     self.model = agg_model_list[max_idx]
                    
                test_metric_agg, test_loss_agg, inference_time_agg = self.test(self.model, device=torch.device('cuda'))
                print('\n')
                # print(f"Choose cluster {max_idx} and {mid_idx} to aggregate")
                print(f"Test acc of agg model: {test_metric_agg}")
                # true_pred = list(set(attacker_idx) & set(cluster_list[min_idx]))
                true_pred = list(set(attacker_idx) & set(attacker_cluster))
                print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                # if test_metric_1 >= test_metric_2:
                #     self.model = aggregate_model_cluster_1
                #     print('\n')
                #     print("Choose cluster 1 to aggregate")
                #     true_pred = list(set(attacker_idx) & set(cluster_2))
                #     print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                # else: 
                #     self.model = aggregate_model_cluster_2
                #     print('\n')
                #     print("Choose cluster 2 to aggregate")
                #     true_pred = list(set(attacker_idx) & set(cluster_1))
                #     print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                dictionary = {
                "round": round,
                "attacker_idx": attacker_idx.tolist(),
                "cluster 0": cluster_0,
                "cluster 1": cluster_1,
                #   "cluster 2": cluster_2,
                #   "test acc all": test_metric_all,
                #   "test acc cluster 0": test_metric_0,
                #   "test acc cluster 1": test_metric_1,
                #   "test acc cluster 2": test_metric_2,
                "Avg_acc_before_train_cluster_0": Avg_acc_before_train_cluster_0,
                "Avg_acc_before_train_cluster_1": Avg_acc_before_train_cluster_1,
                "Chosen cluster": chosen_cluster,
                "attacker_cluster": attacker_cluster,
                "test acc after agg": test_metric_agg,
                "true prediction attacker": [int(i) for i in true_pred],
                }
                path_ = self.option['algorithm'] + '/' + self.option['agg_algorithm'] + '/' + 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/'.format( len(self.option['attacked_class']), self.option['dirty_rate'][0], self.option['proportion']*50, self.option['num_malicious'])
                listObj = []
                if round != 1:
                    with open(path_ + 'log.json') as fp:
                        listObj = json.load(fp)
                
                listObj.append(dictionary)
                
                with open(path_ + 'log.json', 'w') as json_file:
                    json.dump(listObj, json_file, indent=4)
                
                print(f'Done aggregate at round {self.current_round}')
            else:
                self.model = self.aggregate(models)
                print(f'Done aggregate at round {self.current_round}')
                
        return

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.gradL = None
        self.alpha = option['alpha']

    def train(self, model, device, current_round=0):
        if self.uncertainty == 0:
            print(len(self.train_data.idxs))    
                
            # global parameters
            src_model = copy.deepcopy(model).to(device)
            src_model.freeze_grad()
            
            src_model_params = []
            for param in src_model.parameters():
                
                # src_model_params=param.view(-1) if not len(src_model_params) else torch.cat((src_model_params,param.view(-1)))
                src_model_params.append(param.data.view(-1))
            if self.gradL == None:
            #     # self.gradL = model.zeros_like().to(device)
                self.gradL = []
                for param in src_model.parameters():
                    # self.gradL=torch.zeros_like(param).view(-1).to(device) if not len(self.gradL) else torch.cat((self.gradL,torch.zeros_like(param).view(-1).to(device)))
                    self.gradL.append(torch.zeros_like(param.data).view(-1).to(device))
            
                
            model = model.to(device)
            model.train()
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            peer_grad = []
            for iter in range(self.epochs):
                for batch_idx, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    # l1 = self.calculator.get_loss(model, batch_data, device=device)
                    l1 = self.calculator.get_loss_not_uncertainty(model, batch_data, device=device)
                    l2 = 0
                    l3 = 0
                    
                    # for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                    for idx, pm in enumerate(model.parameters()):
                        pgl = self.gradL[idx]
                        ps = src_model_params[idx]
                        l2 += torch.dot(pgl, pm.data.view(-1))
                        l3 += torch.sum(torch.pow(pm.data.view(-1)-ps,2))
                    print("l1 = ", l1)
                    print("l2 = ", l2)
                    print("l3 = ", l3)
                    loss = l1 - l2 + 0.5 * self.alpha * l3
                    print("loss = ", loss)
                    loss.backward()
                    for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            if iter == 0 and batch_idx == 0:
                                peer_grad.append(params.grad.clone())
                            else:
                                peer_grad[i]+= params.grad.clone()
                    optimizer.step()
            # update grad_L
            # self.gradL = self.gradL - self.alpha * (model-src_model)
            for idx, pm in enumerate( model.parameters()):
                self.gradL[idx] = self.gradL[idx] - self.alpha * (pm.data.view(-1) - src_model_params[idx])
            return peer_grad

