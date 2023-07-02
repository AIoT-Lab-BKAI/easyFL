import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


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
    for _ in range(ppo_epochs):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def build_dnn(input_size, num_layer):
    module_list = []
    for i in range(num_layer):
        output_size = input_size // 2
        module_list += [
            nn.Linear(input_size, output_size//2),
            nn.LayerNorm(output_size),
            nn.Relu()
        ]
        input_size = output_size
    return module_list, input_size

class TransformerEncoder(nn.Module):
    
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        # compress the client last classifer layer to lower dimension
        module_list, output_size = build_dnn(num_inputs, 3)
        module_list += [nn.Linear(output_size, hidden_size)]
        self.encoder1 = nn.Sequential(*module_list)

        # encode the whole client layer to a vector
        self.encoder2 = TransformerEncoder(num_layers=3,
                                            input_dim=hidden_size,
                                            dim_feedforward=2*hidden_size,
                                            num_heads=1)

        module_list, output_size = build_dnn(hidden_size, 2)
        module_list += [nn.Linear(output_size, 1)]
        self.critic = nn.Sequential(*module_list)
        
        module_list, output_size = build_dnn(hidden_size, 2)
        module_list += [nn.Linear(output_size, num_outputs)]
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
        
    def forward(self, x):
        # x: [num_clients, classifier_size]
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = torch.sum(x, dim=1, keepdims=True)
        
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def get_action(self, state):
        with torch.no_grad():
            dist, value = self(state)
        
        action = dist.sample()
        
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
    
    def update(self, next_state, optimizer, ppo_epochs=4, mini_batch_size=5):
        next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        
        ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        self.init_rl()
        return
