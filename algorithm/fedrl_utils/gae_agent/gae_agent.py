import torch
import torch.optim as optim

from algorithm.fedrl_utils.ddpg_agent.buffer import *
from algorithm.fedrl_utils.gae_agent.networks import ActorCritic


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


class gae_agent():
    def __init__(self, num_inputs, num_outputs, hidden_size, device):
        self.model     = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        # self.entropy   = 0
        
        self.count     = 0
        self.k_steps   = 5
        self.device    = device
        return
        
        
    def get_action(self, state, prev_reward):
        self.count += 1
        
        if prev_reward != None:
            self.rewards.append(torch.FloatTensor(prev_reward).unsqueeze(1).to(self.device))
            
        if len(self.log_probs) >= self.k_steps:
            assert len(self.rewards) == len(self.log_probs) == len(self.values) == len(self.masks), "Invalid update"
            self.update(state)
            self.clear_storage()
        
        dist, value = self.model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # self.entropy += dist.entropy().mean()
        
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.masks.append(torch.FloatTensor(0).unsqueeze(1).to(self.device))
        return action
    
    
    def update(self, next_state):
        next_state = torch.FloatTensor(next_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)
        
        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss #- 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
        
    
    def clear_storage(self):
        self.log_probs.pop(0)
        self.values.pop(0)
        self.rewards.pop(0)
        self.masks.pop(0)
        # self.entropy = 0
        return
    