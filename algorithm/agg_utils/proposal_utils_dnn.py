import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import pdb
import math



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.0)
        
def compute_gae(next_value, rewards, masks, values, gamma=0.95, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
         

# def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
#     batch_size = states.size(0)
#     for _ in range(batch_size // mini_batch_size):
#         rand_ids = np.random.randint(0, batch_size, mini_batch_size)
#         for _ in range(10):
#             numbers = list(range(0, states.shape[1]))
#             permutations = [random.sample(numbers, len(numbers)) for _ in range(50)]
            
#             states_ = states[rand_ids, :]
#             actions_ = actions[rand_ids, :]
#             log_probs_ = log_probs[rand_ids, :]
#             returns_= returns[rand_ids, :]
#             advantage_ = advantage[rand_ids, :]

#             list_states = [] 
#             list_actions = []
#             list_log_probs = [] 
#             list_returns = []
#             list_advantage = []

#             for perm in permutations:
#                 list_states.append(states_[:, perm]) 
#                 list_actions.append(actions_[:, perm])
#                 list_log_probs.append(log_probs_[:, perm])
#                 list_returns.append(returns_)
#                 list_advantage.append(advantage_)

#             yield torch.cat(list_states), torch.cat(list_actions), torch.cat(list_log_probs), torch.cat(list_returns), torch.cat(list_advantage)
            

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    losses = []
    cri_losses = []
    act_losses = []
    for _ in range(ppo_epochs):
        epochs_loss = []
        cri_loss = []
        act_loss = []
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            
            epochs_loss.append(loss.detach().cpu().item())
            cri_loss.append(critic_loss.detach().cpu().item())
            act_loss.append(actor_loss.detach().cpu().item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(epochs_loss))
        cri_losses.append(np.mean(cri_loss))
        act_losses.append(np.mean(act_loss))
        
        epochs_loss = []
        cri_loss = []
        act_loss = []
    return losses, cri_losses, act_losses

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_clients, epsilon_initial, epsilon_decay, epsilon_min, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.dnn = nn.Sequential(
            nn.Linear(num_inputs[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        dnn_out_size = self._get_dnn_out(num_channel= num_clients, c1 = num_inputs[0], c2 = num_inputs[1])
        self.policy = nn.Sequential(
            nn.Linear(dnn_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
            nn.Tanh()
        )

        self.value = nn.Sequential(
            nn.Linear(dnn_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.clip_param = 0.8

        self.log_std = torch.ones(1, num_outputs) * std
        self.log_std = nn.Parameter(torch.clamp(self.log_std, np.log(0.001), np.log(0.1)))
        
        self.apply(init_weights)
        self.init_rl()

        self.bool = True
        return
    
    def _get_dnn_out(self, num_channel, c1, c2,):
        o = self.dnn(torch.zeros(1, num_channel , c1, c2))
        # print(o.size())
        return int(np.prod(o.size()))
    
    def init_rl(self):
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropy   = 0
        return

    def critic(self, x):
        x = self.dnn(x)
        if self.bool == True:
            # print(x.shape)
            # print("OUTPUT:", x)
            self.bool = False
        x = x.view(x.shape[0], -1)
        return self.value(x)

    def actor(self, x):
        x = self.dnn(x)
        x = x.view(x.shape[0], -1)
        return self.policy(x)

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def epsilon_greedy_action(self, dist, fedavg_action, epsilon, device = None):
        # Hàm này trả về hành động dựa trên xác suất được cung cấp và giá trị epsilon.
        val = np.random.uniform()
        if val < 2 * epsilon:
            if val < epsilon:
                print("+++RANDOM+++")
                random_numbers = [np.random.uniform(-1, 1) for _ in range(len(fedavg_action))]
                mu = torch.tensor(random_numbers, device = device, requires_grad = True).unsqueeze(0)
            else:
                print("+++FEDAVG+++")
                action = [math.log(x) for x in fedavg_action]
                mu = torch.tensor(action, device = device, requires_grad = True).unsqueeze(0)
            std  = self.log_std.exp().expand_as(mu)
            return Normal(mu, std)
            # return torch.randn(action.shape[0], action.shape[1], device = action.device)
        else:
            return dist
    
    def get_action(self, state, fedavg_action):
        dist, value = self(state)
        dist = self.epsilon_greedy_action(dist, fedavg_action, self.epsilon, device = value.device)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean().detach()
        
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.states.append(state.detach())
        self.actions.append(action)
        return torch.softmax(action.reshape(-1), dim=0)
    
    def record(self, reward, done=0, device=None):
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
    
    # def update(self, next_state, optimizer, ppo_epochs=20, mini_batch_size=10):
    #     # next_state = torch.FloatTensor(next_state)
    #     _, next_value = self(next_state)
    #     returns = compute_gae(next_value, self.rewards, self.masks, self.values)
    #     # returns, advantages = compute_gae(next_value, self.rewards, self.masks, self.values)

    #     returns   = torch.cat(returns).detach()
    #     log_probs = torch.cat(self.log_probs).detach()
    #     values    = torch.cat(self.values).detach()
    #     states    = torch.cat(self.states)
    #     actions   = torch.cat(self.actions)
    #     advantage = returns - values
    #     # advantage = torch.cat(advantages).detach()
    #     print("returns: ", returns)
    #     print("values: ", values)

    #     mini_batch_size = len(self.states)//4
    #     losses, cri_losses, act_losses = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param = self.clip_param)
    #     print("Update losses:", losses)
    #     print("Update critic losses:", cri_losses)
    #     print("Update actor losses:", act_losses)

    #     self.init_rl()
    #     self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    #     self.clip_param = max(self.clip_param * 0.9, 0.1)
    #     self.bool = True        
    #     return
    
    def get_experience(self, next_state):
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)
        # returns, advantages = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        return states, actions, log_probs, returns, advantage

    def update(self, optimizer, states, actions, log_probs, returns, advantage, ppo_epochs=10, mini_batch_size=10):
        mini_batch_size = len(self.states)//4
        losses, cri_losses, act_losses = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param = self.clip_param)
        print("Update losses:", losses)
        # print("Update critic losses:", cri_losses)
        # print("Update actor losses:", act_losses)

        self.init_rl()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.clip_param = max(self.clip_param * 0.8, 0.1)
        self.bool = True        
        return