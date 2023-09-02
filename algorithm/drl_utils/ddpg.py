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
from torch.utils.data import Dataset, DataLoader


def KL_divergence(teacher_batch_input, student_batch_input, device):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    Author: Nguyen Nang Hung
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.to(device).unsqueeze(1)
    student_batch_input = student_batch_input.to(device).unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm.flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


class StateDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        return
    
    def insert(self, x):
        self.data.append(x)     # 1 x N x M x d
        return
    
    def __getitem__(self, index):
        return self.data[index].flatten(2,3)
    

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
        
        self.state_dataset = StateDataset()
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


    def sp_update(self, device='cuda'):
        optimizer = optim.SGD(self.state_processor.parameters(), lr=1e-3)
        loader = DataLoader(self.state_dataset, batch_size=4, drop_last=False, shuffle=True)
        losses = []
        for epoch in range(8):
            epoch_losses = []
            for state in loader:
                state = state.to(device)                                            # batch x N x (M * d)
                processed_state = self.state_processor(state).transpose(1,2)        # batch x N x M
                correlation_loss = KL_divergence(state, processed_state, device)
                
                optimizer.zero_grad()
                correlation_loss.backward()
                optimizer.step()
                epoch_losses.append(correlation_loss.detach().cpu().item())
            losses.append(np.mean(epoch_losses))
        return np.mean(losses)
    

    def get_action(self, state, prev_reward, done=False):
        state = torch.DoubleTensor(state).unsqueeze(0).cuda()  # current state: 1 x N x M x d
        self.state_dataset.insert(state)
        preprocessed_state = self.state_processor(state)       # process state: 1 x M x N

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