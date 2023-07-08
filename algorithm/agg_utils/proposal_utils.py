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
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

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
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(10, 32, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 32, kernel_size=(1,5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 32, kernel_size=(1,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(1, 10, kernel_size=(1,5), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        #     nn.Conv2d(10, 32, kernel_size=(1,5), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        #     nn.Conv2d(32, 32, kernel_size=(1,5), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        #     nn.Conv2d(32, 32, kernel_size=(1,3), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 3)),
        # )

        
        conv_out_size = self._get_conv_out(num_inputs, 1, num_outputs)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size*num_outputs, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size*num_outputs, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.fc1 = nn.Linear(conv_out_size, hidden_size)
        self.leakyRelu = nn.LeakyReLU(0.01)
        self.attention = nn.MultiheadAttention(embed_dim = hidden_size, num_heads = 1)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        self.init_rl()
        return
    
    def _get_conv_out(self, shape, num_channel, num_out):
        o = self.conv(torch.zeros(1, num_channel , *shape))
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
    
    def reinit_rl(self):
        del self.log_probs[0]
        del self.values[0]
        del self.states[0]
        del self.actions[0]
        del self.rewards[0]
        del self.masks[0]
        return

    def critic(self, _x):
        x = _x.view(-1, 1, _x.shape[2], _x.shape[3])
        x = self.conv(x)
        x = x.view(_x.shape[0], _x.shape[1], -1)

        q = self.leakyRelu(self.fc1(x))
        k = self.leakyRelu(self.fc1(x))
        v = self.leakyRelu(self.fc1(x))
        attn_output, attn_output_weights = self.attention(q,k,v)

        ## x: (batch_size, out_seq_len, hidden_size)
        x = attn_output.view(_x.shape[0], -1)
        return self.value(x)

    def actor(self, _x):
        x = _x.view(-1, 1, _x.shape[2], _x.shape[3])
        x = self.conv(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(_x.shape[0], _x.shape[1], -1)

        q = self.leakyRelu(self.fc1(x))
        k = self.leakyRelu(self.fc1(x))
        v = self.leakyRelu(self.fc1(x))
        attn_output, attn_output_weights = self.attention(q,k,v)

        ## x: (batch_size, out_seq_len, hidden_size)
        x = attn_output.view(_x.shape[0], -1)
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
        self.values.append(value.detach())
        self.states.append(state.detach())
        self.actions.append(action)
        return torch.softmax(action.reshape(-1), dim=0)
    
    def record(self, reward, done=0):
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1))
        return
    
    def update(self, next_state, optimizer, ppo_epochs=50, mini_batch_size=5):
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
        ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        return

    def begin_iter(self, states, actions, mini_batch_size = 5):
        batch_size = states.shape[0]
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :]
        
    def reinit_weight(self, optimizer, num_epochs, states, actions):
        states = torch.cat(states)
        actions = torch.cat(actions).requires_grad_()
        get_loss = nn.MSELoss()
        for _ in range(num_epochs):
            for state, action in self.begin_iter(states, actions):
                dist, value = self(state)
                action_predict = dist.sample()
                # pdb.set_trace()    
                loss = get_loss(action, action_predict)
                optimizer.zero_grad()
                loss.backward()
                print(loss)
                optimizer.step()