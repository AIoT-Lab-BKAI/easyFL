from .mp_fedbase import MPBasicServer, MPBasicClient
from .utils.mps_utils.cnn_mnist import MyModel
    
class MPSBasicServer(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(MPSBasicServer, self).__init__(option, model, clients, test_data)
        self.model = MyModel()
    
class MPSBasicClient(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(MPSBasicClient, self).__init__(option, name, train_data, valid_data)
