import torch
from algorithm.mp_fedbase import MPBasicServer, MPBasicClient
from utils import fmodule
import numpy as np


def convert_to_array(dictionary_list):
    max_key = 0
    for dictionary in dictionary_list:
        for key, value in dictionary.items():
            max_key = max(max_key, key)
    
    output = np.zeros([len(dictionary_list), max_key+1]) # client x sample
    for i in range(len(dictionary_list)):
        dictionary = dictionary_list[i]
        for key, value in dictionary.items():
            output[i][key] += value
        
    return output

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.optim_ratio = None
        self.sample_distribution_array = None
    
    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        insights = [cp["insight"] for cp in packages_received_from_clients]
        return models, insights

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, insights = self.communicate(self.selected_clients, pool)
        if not self.selected_clients: return

        if self.optim_ratio is None:
            self.optim_ratio = self.process_insight(insights)
            
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = self.optim_ratio)
        return


    def run(self):
        super().run()
        with open(f"fedtask/{self.option['task']}/{self.option['task']}_DataMatrix.csv", "w") as file:
            np.savetxt(file, self.sample_distribution_array, delimiter=',', fmt='%.1e')
        return


    def process_insight(self, insights):
        self.sample_distribution_array = convert_to_array(insights)
        return [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.insight = None

    def reply(self, svr_pkg, device):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)

        if self.insight is None:
            self.insight = self.get_insight()
            print(self.insight.keys())

        cpkg = self.pack(model, self.insight)
        return cpkg


    def pack(self, model, insight):
        return {
            "model": model,
            "insight": insight
        }


    def get_insight(self):
        """
        Returns dictionary of labels and number of samples
        Examples:
            {
                '1': 15,
                '2': 20,
                '6': 12
            }
        """
        features = self.train_data.X
        labels = self.train_data.Y
        
        insight_dict = {}
        for sample, target in zip(features, labels):
            if target.item() not in insight_dict.keys():
                insight_dict[target.item()] = 1
            else:
                insight_dict[target.item()] += 1

        return insight_dict