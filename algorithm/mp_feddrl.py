from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.fedrl_utils.ddpg import DDPG_Agent
import numpy as np
import torch

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.ddpg_agent = DDPG_Agent(state_dim= self.clients_per_round * 2, action_dim= self.clients_per_round, hidden_dim=256, device=f"cuda:{self.server_gpu_id}")
        self.prev_reward = None

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients, pool, t)
                
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        if not self.selected_clients:
            return

        observation = {
            "done": 0,
            "losses": [loss * self.client_vols[cid]/self.data_vol for loss, cid in zip(train_losses, self.selected_clients)],
            "n_samples": [self.client_vols[cid] for cid in self.selected_clients]
        }

        priority = self.ddpg_agent.get_action(observation, prev_reward=self.prev_reward)
        self.model = self.aggregate(models, p=priority)

        self.prev_reward = np.mean(train_losses)
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
