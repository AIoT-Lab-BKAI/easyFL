import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from algorithm.agg_utils.transformer import TransformerEncoder

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        
        
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
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
         

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    losses = []
    for _ in range(ppo_epochs):
        epochs_loss = []
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(epochs_loss))
        epochs_loss = []
    return losses

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
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
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
        
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
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
    
    def get_action(self, state):
        with torch.no_grad():
            dist, value = self(state)
        
        action = dist.sample()
        action = torch.softmax(action, dim=-1)
        
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean()

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(state)
        self.actions.append(action)
        return action
    
    def record(self, reward, done=0):
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1))
        return
    
    def update(self, next_state, optimizer, ppo_epochs=10, mini_batch_size=5):
        next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        
        loss = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        print(loss)
        # while(len(self.states) > 50):
        self.init_rl()
        
        return
