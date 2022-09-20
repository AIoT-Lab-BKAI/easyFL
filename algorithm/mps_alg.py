from .mps_fedbase import MPSBasicServer, MPSBasicClient
import torch
import json
import copy
import torch.nn as nn
import torch.multiprocessing as mp

    
def KDR_loss(teacher_batch_input, student_batch_input, device):
    """
    Compute the Knowledge-distillation based KL divergence of 2 batches of layers
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

class Server(MPSBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.records = {}
        self.clusters = [[0,1,2,3,4], [5,6,7,8,9]]
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        
        models = [model.to("cpu") for model in models]
        head_models, head_list = self.communicate_phase_two(models, self.clusters, t, pool)
        
        print("Heads: ", head_list)
        
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [models[cid] for cid in range(len(models)) if cid not in head_list] + head_models
        models = [model.to(device0) for model in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        
        # Feature Extractors are aggregated using all model        
        new_feature_extractor = self.aggregate([model.feature_generator for model in models], p = impact_factors)
        
        # Classifiers are aggregated using the heads' only
        new_classifier = self.aggregate([model.classifier for model in head_models], p = impact_factors)
        
        self.model.update(new_feature_extractor, new_classifier)
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
        cluster = [0,1,2,3,4] is a list of int (id of clients in this cluster)
        """
        return cluster[time_step % len(cluster)]
    
    def communicate_phase_two(self, models, clusters, time_step, pool):
        """
        clusters = [[0,1,2,3,4], [5,6,7,8], [9,10,11], ...]
            -> is a list of int-list
        """    
        zip_list = []
        head_list = []
        
        for cluster in clusters:
            head = self.select_cluster_head(cluster, time_step)
            classifiers = []
            for cid in cluster:
                if cid != head:
                    classifiers.append(copy.deepcopy(models[cid].classifier))
            zip_list.append({"head" : head, "head_model": models[head], "classifiers": classifiers})
            head_list.append(head)
        
        packages_received_from_clients = []
        packages_received_from_clients = pool.map(self.communicate_phase_two_with, zip_list)
        
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return self.unpack_phase_two(packages_received_from_clients), head_list
    
    def communicate_phase_two_with(self, cluster_dict):
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus
        
        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda') # This is only 'cuda' so its can find the propriate cuda id to train
        
        head_id = cluster_dict['head']
        head_model = cluster_dict['head_model']
        classifiers = cluster_dict['classifiers']
        
        svr_pkg = self.pack_phase_two(head_model, classifiers)
        
        if self.clients[head_id].is_drop():
            return None
        return self.clients[head_id].reply_phase_two(svr_pkg, device)
    
    def pack_phase_two(self, head_model, classifiers):
        return {
            "model": head_model,
            "classifiers" : classifiers,
        }
        
    def unpack_phase_two(self, packages_received_from_clients):
        head_models = [cp["model"] for cp in packages_received_from_clients]
        return head_models
    
class Client(MPSBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kd_factor = 1
        
    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()

        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=False)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, 
                                                  lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, src_model, batch_data, device)
                loss.backward()
                optimizer.step()
        return

    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KDR_loss(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss + kl_loss * self.kd_factor
    
    def pack(self, model, loss):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
        }
    
    def pack_phase_two(self, model):
        return {
            "model": model,
        }
        
    def unpack_phase_two(self, received_pkg):
        return received_pkg['model'], received_pkg['classifiers']
    
    def reply_phase_two(self, svr_pkg, device):
        model, classifiers = self.unpack_phase_two(svr_pkg)
        # self.distill(model, classifiers, device)
        cpkg = self.pack_phase_two(model)
        return cpkg
    
    def distill(self, model, classifiers, device):
        model = model.to(device)
        model.train()
        
        classifiers = [classifier.to(device) for classifier in classifiers]
        for classifier in classifiers:
            classifier.freeze_grad()

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
                total_loss = loss + 0.01 * kd_loss
                total_loss.backward()
                optimizer.step()
        return

    def compute_kd_loss(self, model, classifiers, batch_data, device):
        tdata = self.calculator.data_to_device(batch_data, device)    
        true_output, representation = model.pred_and_rep(tdata[0])
        
        mse_loss = 0
        for classifier in classifiers:
            mse_loss += self.mse(true_output, classifier(representation))

        return mse_loss
            
        