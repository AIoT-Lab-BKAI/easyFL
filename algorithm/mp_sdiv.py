from .mp_fedbase import MPBasicServer, MPBasicClient
import torch


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
        sim.append((x.T @ y) / (torch.norm(x) * torch.norm(y)))

    return torch.mean(torch.tensor(sim)), sim[-1]


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.impact_factor = None
        self.thr = 0.75

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        models = [model.to(torch.device(f"cuda:{self.server_gpu_id}")) for model in models]
        
        if not self.selected_clients: 
            return
        
        if (len(self.selected_clients) < len(self.clients)) or (self.impact_factor is None):
            self.impact_factor = self.get_impact_factor(models)

        self.model = self.aggregate(models, p = self.impact_factor)
        return


    @torch.no_grad()
    def get_impact_factor(self, model_list):
        device = torch.device(f"cuda:{self.server_gpu_id}")
        self.model = self.model.to(device)
        models = []
        
        for model in model_list:
            for p, q in zip(model.parameters(), self.model.parameters()):
                p = p - q
            models.append(model)
        
        similarity_matrix = torch.zeros([len(models), len(models)])
        for i in range(len(models)):
            for j in range(len(models)):
                similarity_matrix[i][j], _ = compute_similarity(models[i], models[j])
        
        similarity_matrix = (similarity_matrix - torch.min(similarity_matrix))/(torch.max(similarity_matrix) - torch.min(similarity_matrix))
        similarity_matrix = similarity_matrix > self.thr
        
        impact_factor = 1/torch.sum(similarity_matrix, dim=0)
        return impact_factor.detach().cpu().tolist()


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
