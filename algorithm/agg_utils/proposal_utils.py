import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pdb


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


class ActorCritic(nn.Module):
    def __init__(self, num_input0, num_input1, num_outputs, hidden_size, kernel_size=3, std=0.0):
        super(ActorCritic, self).__init__()
        
        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )
        
        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_outputs),
        # )
        axis_0 = (num_input0 - kernel_size + 1) // 2
        axis_0 = (axis_0 - kernel_size + 1) // 2

        axis_1 = (num_input1 - kernel_size + 1) // 2
        axis_1 = (axis_1 - kernel_size + 1) // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(num_outputs, 32, kernel_size),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size),
            nn.MaxPool2d(2),
        )

        self.critic_decoder = nn.Sequential(
            nn.Linear(64*axis_0*axis_1, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor_decoder = nn.Sequential(
            nn.Linear(64*axis_0*axis_1, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, num_outputs),
        )

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

    def critic(self, x):
        # pdb.set_trace()
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.critic_decoder(x)

    def actor(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.actor_decoder(x)

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def get_action(self, state):
        dist, value = self(state)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean().detach()
        
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.states.append(state.detach())
        self.actions.append(action)
        return torch.softmax(action.reshape(-1), dim=0)
    
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
        if (len(states) > 50): 
            self.init_rl()
        return
