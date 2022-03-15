from algorithm.fedbase import BasicServer, BasicClient
from utils import fmodule

from algorithm.fedrl_utils.ddpg_agent.ddpg import DDPG_Agent
from datetime import datetime


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        K = self.clients_per_round
        
        self.ddpg_agent = DDPG_Agent(state_dim= K * K, action_dim= K * 3, hidden_dim=256)
        self.buff_folder = f"state{K * K * 3}-action{K*3}"

        now = datetime.now()
        dt_string = now.strftime("%d:%m:%Y-%H:%M:%S")
        self.buff_file = dt_string

        self.prev_reward = None
        self.warmup_length = 20


    def unpack(self, packages_received_from_clients):
        
        assert self.clients_per_round == len(packages_received_from_clients), "Wrong at num clients_per_round"

        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return models, train_losses


    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients:
            return

        observation = {
            "done": 0,
            "models": models
        }

        if t > self.warmup_length:
            priority = self.ddpg_agent.get_action(observation, prev_reward=self.prev_reward).tolist()
            self.model = self.aggregate(models, p=priority)
            fedrl_test_acc, _ = self.test(model=self.model)
            self.prev_reward = fedrl_test_acc
        else:
            self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        
        return

    
    def run(self):
        super().run()
        self.ddpg_agent.dump_buffer(f"algorithm/fedrl_utils/buffers/{self.buff_folder}", self.buff_file)
        return
    
    
    def aggregate(self, models, p=...):
        sump = sum(p)
        p = [pk/sump for pk in p]
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def pack(self, model, loss):
        return {
            "model" : model,
            "train_loss": loss
        }

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg
