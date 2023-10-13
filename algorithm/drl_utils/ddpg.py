# from utils import *
from algorithm.drl_utils.networks import *
from algorithm.drl_utils.policy import *
from algorithm.drl_utils.buffer import *
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
        self.data.append(x)     # N x M x d
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]   # N x M x d
    

class DDPG_Agent(nn.Module):
    def __init__(
        self,
        state_dim=(100, 10, 128),   # (N x M x d)
        action_dim=10,              # K
        hidden_dim=128,
        value_lr=1e-3,
        policy_lr=1e-3,
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

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.state_processor = StateProcessor(state_dim, hidden_dim).cuda()
        self.state_processor_frozen = False

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
        self.total_reward = 0
        return
    
    
    def compute_loss(self, buffer: ReplayBuffer, gamma=0.001, min_value=-np.inf, max_value=np.inf):
        state, action, reward, next_state, done = buffer.sample(self.batch_size)

        state = torch.Tensor(state).squeeze().cuda()
        next_state = torch.Tensor(next_state).squeeze().cuda()
        action = torch.Tensor(action).squeeze().cuda()
        reward = torch.Tensor(reward).cuda()
        done = torch.Tensor(np.float32(done)).cuda()

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = 1/(policy_loss.mean() + 0.001)
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())

        expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action).squeeze()
        value_loss = self.value_criterion(value, expected_value)
        return policy_loss, value_loss
    

    def ddpg_update(self, gamma=0.001, min_value=-np.inf, max_value=np.inf, soft_tau=0.001):
        # for epoch in range(5):
        total_policy_loss, total_value_loss = self.compute_loss(self.replay_buffer, gamma, min_value, max_value)

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

        return total_policy_loss.detach().cpu().item(), total_value_loss.detach().cpu().item()


    def sp_update(self, device='cuda', update=True):
        optimizer = optim.SGD(self.state_processor.parameters(), lr=1e-3)
        loader = DataLoader(self.state_dataset, batch_size=4, drop_last=False, shuffle=True)
        losses = []
        for epoch in range(8):
            epoch_losses = []
            for batch_state in loader:
                batch_state = batch_state.clone().detach().to(device)                           # batch x N x M x d
                batch_processed_state = self.state_processor(batch_state).transpose(1,2)        # batch x N x M
                correlation_loss = 0
                for state, pstate in zip(batch_state, batch_processed_state):
                    correlation_loss += KL_divergence(state.flatten(1,2), pstate, device)
                
                optimizer.zero_grad()
                correlation_loss.backward()                                                     # type:ignore
                if update:
                    optimizer.step()
                    
                epoch_losses.append(correlation_loss.detach().cpu().item())                     # type:ignore
            losses.append(np.mean(epoch_losses))
        return np.mean(losses)
    

    def get_action(self, state, prev_reward, done=False, log=False):
        state = torch.Tensor(state).unsqueeze(0)                        # current state: 1 x N x M x d
        self.state_dataset.insert(state.squeeze(0).cpu())               # put into dataset: N x M x d
        preprocessed_state = self.state_processor(state.cuda())         # process state: 1 x M x N

        if prev_reward is not None:
            self.memory.update(r=prev_reward)
            self.total_reward += prev_reward

        action = self.policy_net(preprocessed_state).flatten()
        # action = self.ou_noise.get_action(action, self.step)
        self.memory.act(preprocessed_state.detach().cpu(), action.detach().cpu())

        if self.memory.get_last_record():
            s, a, r, s_next = self.memory.get_last_record()    # type: ignore
            self.replay_buffer.push(s, a, r, s_next, done)

            if len(self.replay_buffer) >= self.batch_size:
                pl, vl = self.ddpg_update()
                sp_mean = -1 # self.sp_update(update= not self.state_processor_frozen)
                
                if log:
                    wandb.log({"agent/policy_loss": pl, "agent/value_loss": vl,
                               "agent/stateprocessor_loss": sp_mean, "agent/total_reward": self.total_reward}, self.step)
                    
        self.step += 1
        return action


    def load_models(self, path):
        if Path(path).exists():
            try:
                full_path = os.path.join(path, "StateProcessor.pt")
                self.state_processor.load_state_dict(torch.load(full_path))
                print(f"Successfully loaded StateProcessor from {full_path}")
            except:
                print("Failed to load StateProcessor.")
            
            # Policy
            try:
                full_path = os.path.join(path, "MetaPolicyNet.pt")
                self.policy_net.load_state_dict(torch.load(full_path))
                print(f"Successfully loaded MetaPolicyNet from {full_path}")
            except:
                print("Failed to load MetaPolicyNet.")
            
            # Target Policy
            try:
                full_path = os.path.join(path, "MetaTargetPolicyNet.pt")
                self.target_policy_net.load_state_dict(torch.load(full_path))
                print(f"Successfully loaded MetaPolicyNet from {full_path}")
            except:
                print("Failed to load MetaTargetPolicyNet.")
            
            # Value
            try:
                full_path = os.path.join(path, "MetaValueNet.pt")
                self.value_net.load_state_dict(torch.load(full_path))
                print(f"Successfully loaded MetaValueNet from {full_path}")
            except:
                print("Failed to load MetaValueNet.")
            
            # Target Value
            try:
                full_path = os.path.join(path, "MetaTargetValueNet.pt")
                self.target_value_net.load_state_dict(torch.load(full_path))
                print(f"Successfully loaded MetaTargetValueNet from {full_path}")
            except:
                print("Failed to load MetaTargetValueNet.")
        else:
            print(f"{path} does not exist!")
        return
    
    
    def save_models(self, path):
        if not Path(path).exists():
            os.makedirs(path)
        
        self.state_processor.zero_grad()
        self.policy_net.zero_grad()
        self.value_net.zero_grad()
        
        torch.save(self.state_processor.state_dict(), os.path.join(path, "StateProcessor.pt"))
        torch.save(self.policy_net.state_dict(), os.path.join(path, "PolicyNet.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(path, "ValueNet.pt"))
        torch.save(self.target_policy_net.state_dict(), os.path.join(path, "ValueNet.pt"))
        torch.save(self.target_value_net.state_dict(), os.path.join(path, "ValueNet.pt"))
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
                    print(f"\tLoading {filename}... ", end="")
                    old_buffer = ReplayBuffer(capacity=0)
                    if old_buffer.load(path, filename):
                        count += 1
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
    
