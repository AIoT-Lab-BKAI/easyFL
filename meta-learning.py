from algorithm.drl_utils.networks import ValueNetwork, PolicyNetwork
from algorithm.drl_utils.buffer import ReplayBuffer
from copy import deepcopy
from tqdm import tqdm
from torch.autograd import grad

import argparse
import torch
import os
import numpy as np
import torch.nn.functional as F


hidden_dim=128
representation_dim=256
nclass=10
alpha=1e-2
beta=1e-3
soft_tau=0.01
task_epochs=8
batch_size=16
outer_epochs=32


def read_option():
    parser = argparse.ArgumentParser()
    # journal version
    parser.add_argument('--path', help="Folder that contain \"/buffers/\" and \"/models/\"", type=str, default="./storage")
    parser.add_argument('--load_agent', help="If >= 1, then load the pretrained agent from the folder \"storage_path/models\"", type=int, default=1)
    parser.add_argument('--save_agent', help="If >= 1, then save the pretrained agent into the folder \"storage_path/models\"", type=int, default=0)    
    parser.add_argument('--nclient', help="The total number of clients", type=int, default=100)
    parser.add_argument('--clients_per_round', help="The number of participants per round", type=int, default=10)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option


def load_models(path, state_dim, action_dim):
    value_net = ValueNetwork(state_dim, action_dim, hidden_dim).cuda()
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).cuda()
    
    full_path = os.path.join(path, "PolicyNet.pt")
    policy_net.load_state_dict(torch.load(os.path.join(path, "PolicyNet.pt")))
    print(f"Successfully loaded PolicyNet from {full_path}")
    
    full_path = os.path.join(path, "ValueNet.pt")
    value_net.load_state_dict(torch.load(os.path.join(path, "ValueNet.pt")))
    print(f"Successfully loaded ValueNet from {full_path}")
    
    target_value_net = deepcopy(value_net)
    target_policy_net = deepcopy(policy_net)
    
    return value_net, policy_net, target_value_net, target_policy_net


def load_buffers(path):
    res = []
    count = 0
    for filename in os.listdir(path):
        print(f"\tLoading {filename}... ", end="")
        buffer = ReplayBuffer(capacity=0)
        if buffer.load(path, filename):
            count += 1
            res.append(buffer)
            print(f"length: {len(buffer)} - SUCCEEDED.")
    
    print(f"Finished loading {count} buffers.")
    return res


def compute_loss(value_net, policy_net, target_value_net, target_policy_net, buffer: ReplayBuffer, gamma=0.9, min_value=-np.inf, max_value=np.inf):
        state, action, reward, next_state, done = buffer.sample(batch_size)

        state = torch.Tensor(state).squeeze().cuda()
        next_state = torch.Tensor(next_state).squeeze().cuda()
        action = torch.Tensor(action).squeeze().cuda()
        reward = torch.Tensor(reward).cuda()
        done = torch.Tensor(np.float32(done)).cuda()

        policy_loss = value_net(state, policy_net(state))
        policy_loss = - policy_loss.mean()
        next_action = target_policy_net(next_state)
        target_value = target_value_net(next_state, next_action.detach())

        expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = value_net(state, action).squeeze()
        value_loss = F.mse_loss(value, expected_value)
        return policy_loss, value_loss
    

def nth_derivative(f, wrt, n):
    grads = []
    for i in range(n):
        grads = grad(f, wrt, create_graph=True) # type: ignore
        f = grads[0].sum() + grads[1].sum()
    return grads


if __name__ == "__main__":
    option = read_option()
    state_dim=(option['nclient'], nclass, representation_dim)
    action_dim=option['clients_per_round']
    
    value_net, policy_net, target_value_net, target_policy_net = load_models(os.path.join(option['path'], "models"), state_dim, action_dim)
    buffers = load_buffers(os.path.join(option['path'], "buffers"))
    
    for e in tqdm(range(outer_epochs)):
        # epoch_vl = []
        # epoch_pl = []
        
        ast_value_net = deepcopy(value_net)
        ast_policy_net = deepcopy(policy_net)
        
        for task_buffer in tqdm(buffers, leave=False):
            task_value_net = deepcopy(value_net)
            task_policy_net = deepcopy(policy_net)
            
            # per_task_training_pls = []
            # per_task_training_vls = []
            for _ in tqdm(range(task_epochs), leave=False):
                pl, vl = compute_loss(task_value_net, task_policy_net, target_value_net, target_policy_net, task_buffer)
                # per_task_training_pls.append(pl.detach().cpu().item())
                # per_task_training_vls.append(vl.detach().cpu().item())
                
                # First order derivative dL/dtheta
                task_value_grad = nth_derivative(vl, list(task_value_net.parameters()), n=1)
                for param, grad_param in zip(task_value_net.parameters(), task_value_grad):
                    param.data.copy_(param.data - alpha * grad_param)
                
                task_policy_grad = nth_derivative(pl, list(task_policy_net.parameters()), n=1)
                for param, grad_param in zip(task_policy_net.parameters(), task_policy_grad):
                    param.data.copy_(param.data - alpha * grad_param)
                    
                # Second derivative d2L/d(theta)^2
                task_value_sec_order_grad = nth_derivative(vl, list(task_value_net.parameters()), n=2)
                task_policy_sec_order_grad = nth_derivative(pl, list(task_policy_net.parameters()), n=2)
                
                # First order derivative w.r.t the updated parameters (theta')
                pl_prime, vl_prime = compute_loss(task_value_net, task_policy_net, target_value_net, target_policy_net, task_buffer)
                task_value_prime_grad = nth_derivative(vl_prime, list(task_value_net.parameters()), n=1)
                task_policy_prime_grad = nth_derivative(pl_prime, list(task_policy_net.parameters()), n=1)
                
                # Accumulate derivatives into theta*
                for ast_value_param, ast_policy_param, task_value_prime_grad_param, task_policy_prime_grad_param, task_value_sec_order_grad_param, task_policy_sec_order_grad_param in zip(ast_value_net.parameters(), ast_policy_net.parameters(), task_value_prime_grad, task_policy_prime_grad, task_value_sec_order_grad, task_policy_sec_order_grad):
                    ast_value_param.data.copy_(ast_value_param.data - beta * (1 + alpha * task_value_sec_order_grad_param.data) * task_value_prime_grad_param.data)
                    ast_policy_param.data.copy_(ast_policy_param.data - beta * (1 + alpha * task_policy_sec_order_grad_param.data) * task_policy_prime_grad_param.data)

            # Update target network
            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

            for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
            
            # Update theta <- theta*
            for ast_param, param in zip(ast_value_net.parameters(), value_net.parameters()):
                param.data.copy_(ast_param.data)
                
            for ast_param, param in zip(ast_policy_net.parameters(), policy_net.parameters()):
                param.data.copy_(ast_param.data)
            
    