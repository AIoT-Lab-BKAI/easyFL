from torch.distributions import Normal
import torch.nn as nn
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.01):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        
        self.log_std = nn.Parameter(torch.ones(1, num_outputs).squeeze() * std)
        
        self.apply(init_weights)
        
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        
        if torch.isnan(mu).any():
            print("Mu is nan")
            exit(0)
            
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value