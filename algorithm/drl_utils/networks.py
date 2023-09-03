import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class StateProcessor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        """
        For example: state_dim = (100, 10, 128)
        """
        super().__init__()
        _, _, d = state_dim   # 100, 10, 128
        self.linear1 = nn.Linear(d, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
        return
    
    def forward(self, state):
        """
        The state is expected to have the dimension of (batch x N x M x d)
        """
        out = self.linear1(state)               # batch x N x M x hidden_dim
        out = F.relu(out)
        
        out = self.linear2(out).squeeze(-1)     # batch x N x M 
        out = F.relu(out)
        
        return out.transpose(1,2)               # batch x M x N

# self.state_processor = StateProcessor(state_dim=(N, M, d), hidden_dim=128) # output = batch x M x N

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
        N, M, _ = state_dim   # 100, 10, 128
        
        # Input: batch x M x N
        self.linear1 = nn.Linear(N, hidden_dim)
        # Output: batch x M x hidden
        
        self.linear2 = nn.Linear(hidden_dim, 1)
        # Output: batch x M x 1 ---squeeze(-1)---> batch x M 
        
        self.linear3 = nn.Linear(M + action_dim, hidden_dim)
        # Output: batch x hidden
        
        self.linear4 = nn.Linear(hidden_dim, 1)
        # Output: batch x 1

        self.apply(init_weights)
        return
    
    def forward(self, preprocessed_state, action):
        x = F.relu(self.linear1(preprocessed_state))
        x = F.relu(self.linear2(x)).squeeze(-1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network: f(state) -> action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        N, M, _ = state_dim   # 100, 10, 128
        
        # Input: batch x M x N
        self.linear1 = nn.Linear(N, hidden_dim)
        # Output: batch x M x hidden
        
        self.linear2 = nn.Linear(hidden_dim, 1)
        # Output: batch x M x 1 ---squeeze(-1)---> batch x M 
        
        self.linear3 = nn.Linear(M, action_dim)
        # Output: batch x action_dim
        
        self.apply(init_weights)
        return

    def forward(self, preprocessed_state):
        x = F.relu(self.linear1(preprocessed_state))
        x = F.relu(self.linear2(x)).squeeze(-1)
        x = F.softmax(self.linear3(x), dim=1)
        return x
