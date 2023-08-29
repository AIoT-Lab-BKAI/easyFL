import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class StateProcessor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        """
        For example: state_dim = (100, 10, 128)
        """
        super().__init__()
        N, M, d = state_dim   # 100, 10, 128
        self.linear1 = nn.Linear(d, hidden_dim)
        self.linear2 = nn.Linear(N, 1)
        self.apply(init_weights)
        return
    
    def forward(self, state):
        """
        The state is expected to have the dimension of (N x M x d)
        """
        out = self.linear1(state).squeeze()     # N x M x hidden_dim
        out = F.relu(out)
        out = out.transpose(0,2)                # hidden_dim x M x N
        
        out = self.linear2(out).squeeze()       # hidden_dim x M
        out = F.relu(out)
        return out.flatten()                    # hidden_dim * M

# self.state_processor = StateProcessor(state_dim=(N, M, d), hidden_dim=128) # output = hidden_dim * M

class ValueNetwork(nn.Module): 
    """
    Value network: f(state, action) -> goodness
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        state_dim = N x M x d
            where 
                N is the number of client
                M is the number of classes
                d is the dimension of the representation (dim of output of the feature extractor)
        
        action_dim = K
            where 
                K is the number of participants per communication round
        """
        super(ValueNetwork, self).__init__()
        N, M, d = state_dim   # 100, 10, 128
        # self.state_processor = StateProcessor(state_dim=state_dim, hidden_dim=hidden_dim) # output = hidden_dim * M
        
        self.linear1 = nn.Linear(hidden_dim * M + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)
        return
    
    def forward(self, preprocessed_state, action):
        x = torch.cat([preprocessed_state, action])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network: f(state) -> action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        N, M, d = state_dim   # 100, 10, 128
        self.linear1 = nn.Linear(hidden_dim * M, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)
        return

    def forward(self, preprocessed_state):
        x = F.relu(self.linear1(preprocessed_state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
