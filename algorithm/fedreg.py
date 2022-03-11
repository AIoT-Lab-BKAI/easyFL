from algorithm.fedbase import BasicServer, BasicClient
from multiprocessing import Pool as ThreadPool
import torch
import copy


def compute_similarity(a, b):
    """
    Parameters:
        a, b [torch.nn.Module]
    Returns:
        sum of pair-wise similarity between layers of a and b
    """
    sim = 0
    for layer_a, layer_b in zip(a.parameters(), b.parameters()):
        x, y = torch.flatten(layer_a), torch.flatten(layer_b)
        sim += (x.T @ y) / (torch.norm(x) * torch.norm(y))
    return sim


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.rival_list = None
        self.rival_thr = 1
    
    
    def communicate(self, selected_clients):
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
        svr_pkg = self.pack(client_id)
        if self.clients[client_id].is_drop(): 
            return None
        return self.clients[client_id].reply(svr_pkg)
    
    
    def pack(self, client_id):
        return {
            "model" : copy.deepcopy(self.model),
            "rival_models": self.get_rival_of(client_id)
        }

    
    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        
        self.get_rival_list(models)
        
        if not self.selected_clients:
            return
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return

    
    def get_rival_list(self, model_list):
        """
        Considering model a and model b
        If similarity(a, b) < threshold then a is rival of b
        
        Note: this threshold must be somewhat decayable
        """
        similarity_matrix = torch.zeros([len(model_list), len(model_list)])
        for i in range(len(model_list)):
            for j in range(len(model_list)):
                similarity_matrix[i][j] = compute_similarity(model_list[i], model_list[j])
        
        self.rival_list = []
        for i in range(len(model_list)):
            rival = []
            for j in range(len(model_list)):
                if similarity_matrix[i][j] <= self.rival_thr:
                    rival.append(j)
            self.rival_list.append(rival)
        return
    
    
    def get_rival_of(self, client_id):
        i = self.selected_clients.index(client_id)
        return self.rival_list[i]
    

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


    def unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['rival_models']


    def reply(self, svr_pkg):
        model, rival_list = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model, rival_list)
        cpkg = self.pack(model, loss)
        return cpkg


    def train(self, model, rival_list):
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                
                divergence_loss = 0
                for rival in rival_list:
                    for pm, ps in zip(model.parameters(), rival.parameters()):
                        divergence_loss += torch.sum(torch.pow(pm-ps,2))
                
                loss = self.calculator.get_loss(model, batch_data) + 0.05 * divergence_loss
                loss.backward()
                optimizer.step()
        return