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
         

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.05):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            state = state.transpose(0,1)
            action = action.view(-1, 1)
            # pdb.set_trace()
            
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            new_log_probs = new_log_probs.view(old_log_probs.shape[0], -1)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, kernel_size=3, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.conv_value = nn.Sequential(
            nn.Conv2d(num_outputs, 32, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 64, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(64, 64, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
        
        conv_value_out_size = self._get_conv_out(num_inputs, num_channel=num_outputs, type=1)
        
        self.value = nn.Sequential(
            nn.Linear(conv_value_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.conv_policy = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 64, kernel_size=(1,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(64, 64, kernel_size=(1,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            # nn.Conv2d(128, 128, kernel_size=(1,3), stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d((1, 3)),
        )
        
        conv_policy_out_size = self._get_conv_out(num_inputs, num_channel=1, type=2)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, 1) * std)
        
        self.apply(init_weights)
        self.init_rl()
        return
    
    def _get_conv_out(self, shape, num_channel, type):
        if type == 1:
            o = self.conv_value(torch.zeros(1, num_channel , *shape))
        else:
            o = self.conv_policy(torch.zeros(1, num_channel , *shape))
        print(o.size())
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
    
    def reinit_rl(self):
        del self.log_probs[0]
        del self.values[0]
        del self.states[0]
        del self.actions[0]
        del self.rewards[0]
        del self.masks[0]
        return

    def critic(self, x):
        x = x.transpose(0,1)
        x = self.conv_value(x)
        x = x.view(x.shape[0], -1)
        return self.value(x)

    def actor(self, x):
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        x = self.conv_policy(x)
        x = x.view(x.shape[0], -1)
        return self.policy(x)

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def get_action(self, state, action=None):
        dist, value = self(state)
        if action == None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        self.entropy += dist.entropy().mean().detach()
        
        self.log_probs.append(log_prob.transpose(0,1).detach())
        self.values.append(value.transpose(0,1).detach())
        self.states.append(state.transpose(0,1).detach())
        self.actions.append(action.transpose(0,1))
        # pdb.set_trace()
        return torch.softmax(action.reshape(-1), dim=0)
    
    def record(self, reward, done=0):
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1))
        return
    
    def update(self, next_state, optimizer, ppo_epochs=200, mini_batch_size=15):
        next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        
        while (len(self.states) > 60): 
            self.reinit_rl()

        mini_batch_size = len(self.states)//4
        ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        # self.init_rl()
        return

    def begin_iter(self, states, actions, mini_batch_size = 5):
        batch_size = states.shape[0]
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :]
        
    def reinit_weight(self, optimizer, num_epochs, states, actions):
        states = torch.cat(states)
        actions = torch.cat(actions).requires_grad_()
        get_loss = nn.Entro()
        for _ in range(num_epochs):
            for state, action in self.begin_iter(states, actions):
                state = state.transpose(0, 1)
                action = action.view(-1, 1)
                
                dist, value = self(state)
                action_predict = dist.sample()   
                loss = get_loss(action, action_predict)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()