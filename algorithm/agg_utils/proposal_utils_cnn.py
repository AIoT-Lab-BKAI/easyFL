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


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage, num_output):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        for _ in range(0, 50):
            indices = torch.randperm(num_output)
            states = states[:, indices]
            actions = actions[:, indices]
            log_probs = log_probs[:, indices]
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
         

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, num_ouput, clip_param=0.2):
    losses = []
    for _ in range(ppo_epochs):
        epochs_loss = []
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, num_ouput):
            state = state.transpose(0,1)
            # action = action.view(-1, 1)
            
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            # new_log_probs = new_log_probs.view(old_log_probs.shape[0], -1)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            
            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy
            epochs_loss.append(loss.detach().cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses.append(np.mean(epochs_loss))
        epochs_loss = []
    return losses

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
        
        conv_value_out_size = self._get_conv_out(num_inputs, num_channel=num_outputs, type=1, num_out=num_outputs)
        
        self.value = nn.Sequential(
            nn.Linear(conv_value_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
        
        conv_policy_out_size = self._get_conv_out(num_inputs, num_channel=1, type=2, num_out=num_outputs)
        
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        self.log_std = nn.Parameter(torch.ones(1, 1) * std)
        
        self.apply(init_weights)
        self.init_rl()
        return
    
    def _get_conv_out(self, shape, num_channel, type, num_out):
        if type == 1:
            o = self.conv_value(torch.zeros(1, num_channel , *shape))
            return int(np.prod(o.size()))
        else:
            o = self.conv_policy(torch.zeros(1, num_channel , *shape))
            return int(np.prod(o.size()) * num_out)
        # print(o.size())
    
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

    def actor(self, _x):
        x = _x.reshape(-1, 1, _x.shape[2], _x.shape[3])
        x = self.conv_policy(x)
        x = x.reshape(_x.shape[1], -1)
        return self.policy(x)

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
        self.values.append(value.transpose(0,1).detach())
        self.states.append(state.transpose(0,1).detach())
        self.actions.append(action)
        # pdb.set_trace()
        return torch.softmax(action.reshape(-1), dim=0)
    
    def record(self, reward, done=0, device=None):
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
        return
    
    def update(self, next_state, optimizer, ppo_epochs=20, mini_batch_size=15):
        # next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        # pdb.set_trace()

        print("returns: ", returns)
        print("values: ", values)
        
        mini_batch_size = len(self.states)
        losses = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, num_ouput = 10)
        print("Update losses:", losses)
        self.init_rl()
        return