# from utils import *
from networks import *
from policy import *
from buffer import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from copy import deepcopy
import wandb
from pathlib import Path
import os

class DDPG_Agent(nn.Module):
    def __init__(
        self,
        state_dim=(100, 10, 128),   # (N x M x d)
        action_dim=10,              # K
        hidden_dim=128,
        value_lr=1e-3,
        policy_lr=1e-4,
        replay_buffer_size=1000,
        batch_size=4,
        log_dir="./log/epochs",
    ):
        super(DDPG_Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = []
        self.states = []
        self.log_probs = []
        self.values = []
        self.rewards = []  # rewards for each episode

        self.ou_noise = OUNoise(num_actions=action_dim, action_min_val=0, action_max_val=1)

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).cuda().double()
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).cuda().double()
        self.state_processor = StateProcessor(state_dim, hidden_dim).cuda().double()

        self.target_value_net = deepcopy(self.value_net)
        self.target_policy_net = deepcopy(self.policy_net)

        # store all the (s, a, s', r) during the transition process
        self.memory = Memory()
        # replay buffer used for main training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        # replay buffer of old episodes
        self.old_buffers = []

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()
        self.step = 0
        self.batch_size = batch_size
        self.log_dir = log_dir
        return
    
    
    def compute_loss(self, buffer: ReplayBuffer, gamma=0.9, min_value=-np.inf, max_value=np.inf):
        state, action, reward, next_state, done = buffer.sample(self.batch_size)

        state = torch.DoubleTensor(state).squeeze().cuda()
        next_state = torch.DoubleTensor(next_state).squeeze().cuda()
        action = torch.DoubleTensor(action).squeeze().cuda()
        reward = torch.DoubleTensor(reward).cuda()
        done = torch.DoubleTensor(np.float32(done)).cuda()

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = - policy_loss.mean()
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())

        expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action).squeeze()
        value_loss = self.value_criterion(value, expected_value)
        return policy_loss, value_loss
    

    def ddpg_update(self, gamma=0.9, min_value=-np.inf, max_value=np.inf, soft_tau=2e-2):
        online_policy_loss, online_value_loss = self.compute_loss(self.replay_buffer, gamma, min_value, max_value)
        
        offline_policy_loss = 0
        offline_value_loss = 0
        for old_buffer in self.old_buffers:
            pl, vl = self.compute_loss(old_buffer, gamma, min_value, max_value)
            offline_policy_loss += pl
            offline_value_loss += vl
        
        total_policy_loss = 0.5 * online_policy_loss + 0.5 * offline_policy_loss/len(self.old_buffers)
        total_value_loss = 0.5 * online_value_loss + 0.5 * offline_value_loss/len(self.old_buffers)

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        return


    def get_action(self, state, prev_reward, done=False):
        state = torch.DoubleTensor(state).unsqueeze(0).cuda()  # current state
        preprocessed_state = self.state_processor(state)

        if prev_reward is not None:
            self.memory.update(r=prev_reward)

        action = self.policy_net(preprocessed_state)
        action = self.ou_noise.get_action(action, self.step)
        self.memory.act(state, action)

        if self.memory.get_last_record() is not None:
            s, a, r, s_next = self.memory.get_last_record()
            self.replay_buffer.push(s, a, r, s_next, done)

            if len(self.replay_buffer) >= self.batch_size:
                self.ddpg_update()
            
        self.step += 1
        return action


    def load_models(self, path):
        if Path(path).exists():
            try:
                self.state_processor.load_state_dict(torch.load(os.path.join(path, "StateProcessor.pt")))
            except:
                print("Failed to load StateProcessor.")
            
            try:
                self.policy_net.load_state_dict(torch.load(os.path.join(path, "PolicyNet.pt")))
            except:
                print("Failed to load PolicyNet.")
            
            try:
                self.value_net.load_state_dict(torch.load(os.path.join(path, "ValueNet.pt")))
            except:
                print("Failed to load ValueNet.")
        else:
            print(f"{path} does not exist!")
        return
    
    
    def save_models(self, path):
        if not Path(path).exists():
            os.makedirs(path)
        
        torch.save(self.state_processor.state_dict(), os.path.join(path, "StateProcessor.pt"))
        torch.save(self.policy_net.state_dict(), os.path.join(path, "PolicyNet.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(path, "ValueNet.pt"))
        return
    
    
    def load_buffer(self, path, discard_name):
        """
        Load all the buffer into self.old_buffers
        except for the file whose name contains 'discard_name'
        """
        if Path(path).exists():
            print("Start loading experiences...")
            count = 0
            for filename in os.listdir(path):
                if discard_name not in filename:
                    count += 1
                    print(f"\tLoading {filename}... ", end="")
                    old_buffer = ReplayBuffer(capacity=0)
                    old_buffer.load(path, filename)
                    self.old_buffers.append(old_buffer)
                    print(f"length: {len(old_buffer)} - SUCCEEDED.")
            print(f"Finished loading {count} buffers.")
        else:
            print(f"{path} does not exist!")
        return


    def save_buffer(self, path, name):
        """
        Save all the memory into a file
        """
        self.replay_buffer.save(path, name)
        return
    
# if __name__ == "__main__":
#     max_frames = 12000
#     max_steps = 16
#     frame_idx = 0
#     rewards = []
#     batch_size = 128
#     # Test simulator
#     env = NormalizedActions(gym.make("Pendulum-v0"))
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     hidden_dim = 10
#     agent = DDPG_Agent(
#         state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
#     )
#     ou_noise = OUNoise(env.action_space)
#     reward, done = None, None

#     while frame_idx < max_frames:
#         state = env.reset()
#         ou_noise.reset()
#         # episode_reward = 0

#         for step in range(max_steps):
#             # action = policy_net.get_action(state)
#             # action = ou_noise.get_action(action, step)
#             action = agent.get_action(state, reward, done)
#             next_state, reward, done, _ = env.step(action)

#             # replay_buffer.push(state, action, reward, next_state, done)
#             # if len(replay_buffer) > batch_size:
#             #     ddpg_update(batch_size)

#             state = next_state

#             # episode_reward += reward
#             # frame_idx += 1

#             # if frame_idx % max(1000, max_steps + 1) == 0:
#             #     plot(frame_idx, rewards)

#             # if done:
#             #     break

#         # rewards.append(episode_reward)