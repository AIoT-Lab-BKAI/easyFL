import random
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.cfmtx.cfmtx import cfmtx_test
from algorithm.agg_utils.proposal_utils_dnn import ActorCritic

import torch.nn as nn
import numpy as np
import torch
import copy
import torch.multiprocessing as mp
import os
import utils.fflow as flw
import json
from pathlib import Path
from main import logger

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res


def get_classifier(model):
    modules = get_module_from_model(model)
    penul = modules[-1]._parameters['weight']
    return penul


def KL_divergence(teacher_batch_input, student_batch_input, device):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    Author: Nguyen Nang Hung
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.to(device).unsqueeze(1)
    student_batch_input = student_batch_input.to(device).unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm.flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    sim = []
    for layer_a, layer_b in zip(a.parameters(), b.parameters()):
        x, y = torch.flatten(layer_a), torch.flatten(layer_b)
        sim.append((x.transpose(-1,0) @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        classifier = get_classifier(model)
        self.agent = ActorCritic(num_inputs=classifier.shape, num_outputs=self.clients_per_round, num_clients= self.num_clients, 
                                 epsilon_initial = 0.3, epsilon_decay=0.8, epsilon_min=0.02, hidden_size=256, std=np.log(0.1))
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3) # example
        self.steps = 10
        self.buffer_name = option['buffer']
        self.current_buffer = []
        self.replay_buffer = []
        return
    
    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        pool = mp.Pool(self.num_threads)
        logger.time_start('Total Time Cost')

        modelpath = f"./algorithm/model/model_weights.pth"
        if Path(modelpath).exists():
            self.agent.load_state_dict(torch.load(modelpath))
        
        file_list = os.listdir("./algorithm/data")
        for file_name in file_list:
            if file_name.split('.')[0] != self.buffer_name:
                loaded_array = json.load(open(f"./algorithm/data/{file_name}", 'r'))
                self.replay_buffer += loaded_array
        
        print(f"=====LENGTH OF REPLAY BUFFER: {len(self.replay_buffer)}=====")

        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round, pool)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)

        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        filepath = os.path.join(self.log_folder, self.option['task'], self.option['dataidx_filename']).split('.')[0]
        if not Path(filepath).exists():
            os.system(f"mkdir -p {filepath}")
        logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))

        
        torch.save(self.agent.state_dict(), modelpath)

        with open(f"./algorithm/data/{self.buffer_name}.json", 'w') as file:
            json.dump(self.current_buffer, file)
        

    def iterate(self, t, pool):
        self.selected_clients = sorted(self.sample())
        models, train_losses = self.communicate(self.selected_clients, pool)

        # # Kết hợp các phần tử từ hai danh sách thành một danh sách kết hợp
        # combined_list = list(zip(self.selected_clients, models, train_losses))

        # # Shuffle danh sách kết hợp
        # random.shuffle(combined_list)

        # # Tách các phần tử đã được shuffle thành hai danh sách mới
        # self.selected_clients, models, train_losses = zip(*combined_list)
        print ("Selected client: ",self.selected_clients)
        
        if not self.selected_clients: 
            return
        
        # Get classifiers
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        logger.time_start('Cuda|Compute Delta w')
        self.agent = self.agent.to(device0)
        models = [model.to(device0) - self.model.to(device0) for model in models]

        weight_zero = self.model - self.model
        list_model = []
        for idx in range(self.num_clients):
            if idx in self.selected_clients:
                list_model.append(models[self.selected_clients.index(idx)])
            else:
                list_model.append((weight_zero).to(device0))

        logger.time_end('Cuda|Compute Delta w')
        
        classifiers = [get_classifier(submodel) for submodel in list_model]

        classifiers_update = []

        for clf in classifiers:
            max_value = torch.max(clf)
            min_value = torch.min(clf)
            if max_value != min_value:
                classifiers_update.append((clf - min_value)/(max_value - min_value))
            else:
                classifiers_update.append(clf)

        state = torch.stack(classifiers_update)         # <-- Change to matrix K x d
        state = torch.unsqueeze(state, dim=0)    # <-- Change to matrix K x 1 x d
        
        # Processing
        if t > 0:
            reward = - np.mean(train_losses) - (np.max(train_losses) - np.min(train_losses))
            self.agent.record(reward, device=device0)
            if t%self.steps == 0:
                states, actions, log_probs, returns, advantage = self.agent.get_experience(state)
                for idx in range(len(states)):
                    self.current_buffer.append([states[idx].tolist(), actions[idx].tolist(), log_probs[idx].tolist(), returns[idx].tolist(), advantage[idx].tolist()])
                    # self.current_buffer.append([actions[idx].detach().tolist(), log_probs[idx].detach().tolist(), returns[idx].detach().tolist(), advantage[idx].detach().tolist()])
                    

                rand_ids = np.random.randint(0, len(self.replay_buffer), min(int(len(self.replay_buffer)/2), 100))
                rand_ids2 = np.random.randint(0, len(self.current_buffer), min(len(self.current_buffer), 100))

                experiences = [self.replay_buffer[i] for i in rand_ids]
                cur_experiences = [self.current_buffer[i] for i in rand_ids2]

                experiences += cur_experiences

                states2 = torch.tensor([elm[0] for elm in experiences]).to(device0)
                actions2 = torch.tensor([elm[1] for elm in experiences]).to(device0)
                log_probs2 = torch.tensor([elm[2] for elm in experiences]).to(device0)
                returns2 = torch.tensor([elm[3] for elm in experiences]).to(device0)
                advantage2 = torch.tensor([elm[4] for elm in experiences]).to(device0)

                states    = torch.cat((states, states2))
                actions   = torch.cat((actions, actions2))
                log_probs = torch.cat((log_probs, log_probs2))
                returns   = torch.cat((returns, returns2))
                advantage = torch.cat((advantage, advantage2))

                # self.agent.update(state, self.agent_optimizer) # example
                # print("+++++++", states.shape, actions.shape, log_probs.shape, returns.shape, advantage.shape)
                self.agent.update(self.agent_optimizer, states, actions, log_probs, returns, advantage)
                
        impact_factors = self.agent.get_action(state, fedavg_action = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        print("IMPACT FACTOR", impact_factors)
        logger.time_start('Aggregation')
        self.model = self.model + self.aggregate(models, p = impact_factors)
        logger.time_end('Aggregation')
        return
    

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_fct = option['kd_fct']
        
    def reply(self, svr_pkg, device):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)
        cpkg = self.pack(model, loss)
        return cpkg
        
    def train(self, model, device='cuda'):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
                
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(model, src_model, batch_data, device)
                loss = loss + self.kd_fct * kl_loss
                loss.backward()
                optimizer.step()
        return

    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KL_divergence(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, kl_loss
