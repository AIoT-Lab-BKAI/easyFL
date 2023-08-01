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
        nn.init.constant_(m.bias, 0.001)
        
        
def compute_gae(next_value, rewards, masks, values, gamma=0.9, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# def compute_gae(next_value, rewards, masks, values, gamma=0.5, tau=0.95):
#     values = values + [next_value]
#     gae = 0
#     returns = []
#     advantages = []
#     for step in reversed(range(len(rewards))):
#         td_target = rewards[step] + gamma * values[step + 1] * masks[step] 
#         delta = td_target - values[step]
#         gae = delta + gamma * tau * masks[step] * gae
#         advantages.insert(0, gae)
#         returns.insert(0, td_target)
#     return returns, advantages


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
         

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

            loss = 0.5 * critic_loss + 20 * actor_loss - 0.001 * entropy
            
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
    def __init__(self, num_inputs, num_outputs, epsilon_initial, epsilon_decay, epsilon_min, hidden_size, kernel_size=3, std=0.0):
        super(ActorCritic, self).__init__()
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 10, kernel_size=(1,5), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        #     nn.Conv2d(10, 32, kernel_size=(1,5), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        #     nn.Conv2d(32, 64, kernel_size=(1,3), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1, 4)),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),
        )

        
        conv_out_size = self._get_conv_out(num_inputs, 10, num_outputs)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

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
        x = _x.transpose(1,2)
        # print("\nINIT", x[0, 0, 0:10, 0:5])
        x = self.conv(x)
        # print(x.shape)
        # print("OUTPUT:", x[0,0:2,:, 0:5])
        x = x.view(x.shape[0], -1)
        return self.value(x)

    def actor(self, _x):
        x = _x.transpose(1,2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.policy(x)

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def epsilon_greedy_action(self, dist, epsilon):
        action = dist.sample()
        # Hàm này trả về hành động dựa trên xác suất được cung cấp và giá trị epsilon.
        if np.random.uniform() < epsilon:
            print("+++RANDOM+++")
            return torch.randn(action.shape[0], action.shape[1], device = action.device)
        else:
            return action
    
    def get_action(self, state):
        dist, value = self(state)
        action = self.epsilon_greedy_action(dist, self.epsilon)
        
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
    
    def update(self, next_state, optimizer, ppo_epochs=50, mini_batch_size=10):
        # next_state = torch.FloatTensor(next_state)
        _, next_value = self(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)
        # returns, advantages = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values    = torch.cat(self.values).detach()
        states    = torch.cat(self.states)
        actions   = torch.cat(self.actions)
        advantage = returns - values
        # advantage = torch.cat(advantages).detach()

        print("returns: ", returns)
        print("values: ", values)

        mini_batch_size = len(self.states)//2
        losses, cri_losses, act_losses = ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        print("Update losses:", losses)
        print("Update critic losses:", cri_losses)
        print("Update actor losses:", act_losses)

        # self.init_rl()
        while(len(self.states) > 100):
            self.reinit_rl()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)        
        return