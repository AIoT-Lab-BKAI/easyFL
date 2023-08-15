import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from algorithm.agg_utils.transformer import TransformerEncoder
import numpy as np
import random

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.001)
        
        
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
        for _ in range(10):
            numbers = list(range(0, states.shape[1]))
            permutations = [random.sample(numbers, len(numbers)) for _ in range(50)]
            
            states_ = states[rand_ids, :]
            actions_ = actions[rand_ids, :]
            log_probs_ = log_probs[rand_ids, :]
            returns_= returns[rand_ids, :]
            advantage_ = advantage[rand_ids, :]

            list_states = [] 
            list_actions = []
            list_log_probs = [] 
            list_returns = []
            list_advantage = []

            for perm in permutations:
                list_states.append(states_[:, perm]) 
                list_actions.append(actions_[:, perm])
                list_log_probs.append(log_probs_[:, perm])
                list_returns.append(returns_)
                list_advantage.append(advantage_)

            yield torch.cat(list_states), torch.cat(list_actions), torch.cat(list_log_probs), torch.cat(list_returns), torch.cat(list_advantage)
            
# def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
#     batch_size = states.size(0)
#     for _ in range(batch_size // mini_batch_size):
#         rand_ids = np.random.randint(0, batch_size, mini_batch_size)
#         yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
         

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

def build_dnn(input_size, num_layer):
    module_list = []
    for i in range(num_layer):
        output_size = input_size // 2
        module_list += [
            nn.Linear(input_size, output_size//2),
            nn.LayerNorm(output_size),
            nn.ReLU()
        ]
        input_size = output_size
    return module_list, input_size



class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs,epsilon_initial, epsilon_decay, epsilon_min, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        # compress the client last classifer layer to lower dimension
        encoder1_layerdims = [num_inputs, 2560, 1280, hidden_size]
        encoder1 = []
        for i in range(len(encoder1_layerdims)-1):
            encoder1 += [
                nn.Linear(encoder1_layerdims[i], encoder1_layerdims[i+1]),
                nn.LayerNorm(encoder1_layerdims[i+1]),
                nn.ReLU()
            ]
        self.encoder1 = nn.Sequential(*encoder1)

        # encode the whole client layer to a vector
        self.encoder2 = TransformerEncoder(num_layers=3, input_dim=hidden_size, num_heads=16,  dim_feedforward=2*hidden_size)

        layerdims = [hidden_size, 256, 128, 64, 32]
        module_list = []
        for i in range(len(layerdims)-1):
            module_list += [
                nn.Linear(layerdims[i], layerdims[i+1]),
                nn.LayerNorm(layerdims[i+1]),
                nn.ReLU()
            ]
        module_list += [nn.Linear(layerdims[-1], 1)]
        self.critic = nn.Sequential(*module_list)
        
        layerdims = [hidden_size, 256]
        module_list = []
        for i in range(len(layerdims)-1):
            module_list += [
                nn.Linear(layerdims[i], layerdims[i+1]),
                nn.LayerNorm(layerdims[i+1]),
                nn.ReLU()
            ]
        module_list += [nn.Linear(layerdims[-1], num_outputs)]
        self.actor = nn.Sequential(*module_list)
        
        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.clip_param = 0.6

        self.log_std = torch.ones(1, num_outputs) * std
        self.log_std = nn.Parameter(torch.clamp(self.log_std, np.log(0.001), np.log(0.1)))
        
        self.apply(init_weights)
        self.init_rl()
        return
    
    def init_rl(self):
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropy   = 0
        return
    
    def reset_rl(self):
        del self.log_probs[0]
        del self.values[0]
        del self.states[0]
        del self.actions[0]
        del self.rewards[0]
        del self.masks[0]
        return 
    
    def forward(self, x):
        # x: [num_clients, classifier_size]
        x = self.encoder1(x)
        x = self.encoder2(x)

        x = torch.sum(x, dim=(1))
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
                mu = torch.randn((1, 10), device = device, requires_grad = True)
            else:
                print("+++FEDAVG+++")
                mu = torch.tensor(fedavg_action, device = device, requires_grad = True).unsqueeze(0)
            std   = self.log_std.exp().expand_as(mu)
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
        return
    
    def update(self, next_state, optimizer, ppo_epochs=20, mini_batch_size=5):
        # next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        
        print("returns: ", returns)
        print("values: ", values)

        mini_batch_size = len(self.states)//2
        losses, cri_losses, act_losses = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param = self.clip_param)
        print("Update losses:", losses)
        print("Update critic losses:", cri_losses)
        print("Update actor losses:", act_losses)
        

        self.init_rl()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.clip_param = max(self.clip_param * 0.9, 0.1)

        return
