import torch
import torch.optim as optim

from algorithm.fedrl_utils.ddpg_agent.buffer import *
from algorithm.fedrl_utils.gae_agent.networks import ActorCritic


def compute_gae(next_value, rewards, masks, values, gamma=0.25, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


class gae_agent():
    def __init__(self, state_dim, action_dim, hidden_size, device):
        self.model     = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropy   = 0.0
        
        self.count     = 0
        self.k_steps   = 4
        self.device    = device
        
        self.imitate_loss = 0.0
        self.itm_factor = 1
        self.itm_decay = 0.9
        return
        
        
    def get_action(self, state, prev_reward):
        self.count += 1
        state = state.to(self.device)
        if prev_reward != None:
            print("Last action reward: ", prev_reward)
            self.rewards.append(prev_reward.to(self.device))
            
        if len(self.log_probs) >= self.k_steps:
            assert len(self.rewards) == len(self.log_probs) == len(self.values) == len(self.masks), "Invalid update"
            self.update(state)
            self.clear_storage()
        
        if torch.isnan(state).any():
            print("State contains nan")
            exit(0)
        
        dist, value = self.model(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        entp = dist.entropy().mean()
        
        if torch.isnan(entp).any():
            print("Entropy contains nan")
            print(dist.entropy())
            exit(0)
            
        if torch.isnan(value).any():
            print("Entropy contains nan")
            print(value)
            exit(0)
            
        self.entropy += entp
        
        self.log_probs.append(log_prob.mean())
        self.values.append(value)
        self.masks.append(torch.FloatTensor(1).unsqueeze(1).to(self.device))
        
        exp_action = torch.exp(action)
        action = exp_action/torch.sum(exp_action)
        return action
    
    
    def update(self, next_state):
        # next_state = torch.FloatTensor(next_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)
        
        log_probs = torch.vstack(self.log_probs)
        returns   = torch.vstack(returns).detach()
        values    = torch.vstack(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        if torch.isnan(actor_loss).any():
            print("actor loss is nan")
            print("log prob:", log_probs)
            print("advantage:", advantage)
            exit(0)
        
        if torch.isnan(critic_loss).any():
            print("critic loss is nan")
            print("advantage:", advantage)
            exit(0)
            
        loss = actor_loss + 0.5 * critic_loss + self.itm_factor * self.imitate_loss - 0.001 * self.entropy
        self.itm_factor *= self.itm_decay
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        return
    
    
    def reflex_update(self, action, guidence):
        guidence = torch.Tensor(guidence)
        guidence = guidence/torch.sum(guidence)
        
        action = action.to(self.device)
        guidence = guidence.to(self.device)
        itm = torch.mean(guidence * torch.log(guidence/action))
        
        if torch.isnan(itm).any():
            print("imitate_loss is nan")
            print("action:", action)
            print("guidence:", guidence)
            exit(0)
            
        self.imitate_loss += itm
        return
    
    def clear_storage(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()
        self.imitate_loss = 0.0
        self.entropy = 0.0
        return