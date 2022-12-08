import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, init_w=1e-1):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(init_weights)
        return


    def forward(self, state, action, batch_size):
        state = state.view(batch_size, -1)
        action = action.view(batch_size, -1)

        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, init_w=1e-1):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3i = nn.Linear(hidden_size, num_outputs)
        self.activation = nn.Softmax()

        self.apply(init_weights)
        return
        
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        impact = F.leaky_relu(self.linear3i(x))
        impact = self.activation(impact)
        return impact

    def get_action(self, state):
        action = self.forward(state)
        action = torch.flatten(action)
        return action.detach().cpu()

