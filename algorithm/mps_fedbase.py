from .mp_fedbase import MPBasicServer, MPBasicClient
from .utils.newalgo_utils.cnn_mnist import MyModel
import torch
    
class MPSBasicServer(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(MPSBasicServer, self).__init__(option, model, clients, test_data)
        self.model = MyModel()
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool)
        
        if not self.selected_clients: 
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [model.to(device0) for model in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        new_feature_extractor = self.aggregate([model.feature_generator for model in models], p = impact_factors)
        new_classifier = self.aggregate([model.classifier for model in models], p = impact_factors)
        
        self.model.update(new_feature_extractor, new_classifier)
        return
    
class MPSBasicClient(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(MPSBasicClient, self).__init__(option, name, train_data, valid_data)
