from torch.distributions import Normal
import torch.nn as nn
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.01):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        
        self.log_std = nn.Parameter(torch.ones(1, action_dim).squeeze() * std)
        
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