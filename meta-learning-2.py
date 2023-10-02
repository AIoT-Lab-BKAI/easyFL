from algorithm.drl_utils.networks import ValueNetwork, PolicyNetwork
from algorithm.drl_utils.buffer import ReplayBuffer
from copy import deepcopy
from tqdm import tqdm
from torch.autograd import grad
from pathlib import Path

import argparse
import torch
import os
import numpy as np
import torch.nn.functional as F
import json

def read_option():
    parser = argparse.ArgumentParser()
    # journal version
    parser.add_argument('--path', help="Folder that contain \"/buffers/\" and \"/models/\"", type=str, default="./storage")
    parser.add_argument('--load', help="If 1, then load the models for meta learning, otherwise meta-learning from scratch", type=int, default=1)
    parser.add_argument('--save', help="Folder to save the meta models", type=str, default="./storage/models")
    parser.add_argument('--nclient', help="The total number of clients", type=int, default=100)
    parser.add_argument('--clients_per_round', help="The number of per round participants", type=int, default=10)
    parser.add_argument('--hidden_dim', help="", type=int, default=128)
    parser.add_argument('--representation_dim', help="", type=int, default=256)
    parser.add_argument('--nclass', help="", type=int, default=10)
    parser.add_argument('--alpha', help="", type=float, default="0.001")
    parser.add_argument('--beta', help="", type=float, default="0.001")
    parser.add_argument('--soft_tau', help="", type=float, default="0.001")
    parser.add_argument('--task_epochs', help="", type=int, default=1)
    parser.add_argument('--batch_size', help="", type=int, default=32)
    parser.add_argument('--outer_epochs', help="", type=int, default=5000)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option


def load_models(option, state_dim, action_dim):
    value_net = ValueNetwork(state_dim, action_dim, option['hidden_dim']).cuda()
    policy_net = PolicyNetwork(state_dim, action_dim, option['hidden_dim']).cuda()
    target_value_net = deepcopy(value_net)
    target_policy_net = deepcopy(policy_net)
    
    if option['load']:
        path = os.path.join(option['path'], "models")
        
        try:
            full_path = os.path.join(path, "PolicyNet.pt")
            policy_net.load_state_dict(torch.load(full_path))
            print(f"Successfully loaded PolicyNet from {full_path}")
        except:
            print("Failed to load policy net!")
        
        try:
            full_path = os.path.join(path, "ValueNet.pt")
            value_net.load_state_dict(torch.load(full_path))
            print(f"Successfully loaded ValueNet from {full_path}")
        except:
            print("Failed to load value net!")
        
        try:
            full_path = os.path.join(path, "TargetValueNet.pt")
            target_value_net.load_state_dict(torch.load(full_path))
            print(f"Successfully loaded TargetValueNet from {full_path}")
        except:
            print("Failed to load target value net!")
            
        try:
            full_path = os.path.join(path, "TargetPolicyNet.pt")
            target_policy_net.load_state_dict(torch.load(full_path))
            print(f"Successfully loaded TargetPolicyNet from {full_path}")
        except:
            print("Failed to load target policy net!")
    
    
    return value_net, policy_net, target_value_net, target_policy_net


def load_buffers(path):
    count = 0
    buffer = ReplayBuffer(capacity=0)
    for filename in os.listdir(path):
        print(f"\tLoading {filename}... ", end="")
        if buffer.load(path, filename):
            count += 1
            print(f"length: {len(buffer)} - SUCCEEDED.")
    
    print(f"Finished loading {count} buffers into one buffer of size {len(buffer)}.")
    return buffer


def compute_loss(value_net, policy_net, target_value_net, target_policy_net, batch_size,
                 buffer: ReplayBuffer, gamma=0.9, min_value=-np.inf, max_value=np.inf):
        state, action, reward, next_state, done = buffer.sample(batch_size)

        state = torch.Tensor(state).squeeze().cuda()
        next_state = torch.Tensor(next_state).squeeze().cuda()
        action = torch.Tensor(action).squeeze().cuda()
        reward = torch.Tensor(reward).cuda()
        done = torch.Tensor(np.float32(done)).cuda()

        policy_loss = value_net(state, policy_net(state))
        policy_loss = 1/(policy_loss.mean() + 0.001)
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
    state_dim=(option['nclient'], option['nclass'], option['representation_dim'])
    action_dim=option['clients_per_round']
    
    value_net, policy_net, target_value_net, target_policy_net = load_models(option, state_dim, action_dim)
    buffer = load_buffers(os.path.join(option['path'], "guided"))
    
    epoch_vl = []
    epoch_pl = []
        
    for e in tqdm(range(option['outer_epochs'])):
        ast_value_net = deepcopy(value_net)
        ast_policy_net = deepcopy(policy_net)
        
        all_task_training_pl = []
        all_task_training_vl = []
        
        for _ in range(option['task_epochs']):
            pl, vl = compute_loss(ast_value_net, ast_policy_net, target_value_net, target_policy_net, option['batch_size'], buffer)
            all_task_training_pl.append(pl.detach().cpu().item())
            all_task_training_vl.append(vl.detach().cpu().item())
            
            # First order derivative dL/dtheta
            task_value_grad = nth_derivative(vl, list(ast_value_net.parameters()), n=1)
            for param, grad_param in zip(ast_value_net.parameters(), task_value_grad):
                param.data.copy_(param.data - option['alpha'] * grad_param)
            
            task_policy_grad = nth_derivative(pl, list(ast_policy_net.parameters()), n=1)
            for param, grad_param in zip(ast_policy_net.parameters(), task_policy_grad):
                param.data.copy_(param.data - option['alpha'] * grad_param)
                
            # Second derivative d2L/d(theta)^2
            task_value_sec_order_grad = nth_derivative(vl, list(ast_value_net.parameters()), n=2)
            task_policy_sec_order_grad = nth_derivative(pl, list(ast_policy_net.parameters()), n=2)
            
            # First order derivative w.r.t the updated parameters (theta')
            pl_prime, vl_prime = compute_loss(ast_value_net, ast_policy_net, target_value_net, target_policy_net, option['batch_size'], buffer)
            task_value_prime_grad = nth_derivative(vl_prime, list(ast_value_net.parameters()), n=1)
            task_policy_prime_grad = nth_derivative(pl_prime, list(ast_policy_net.parameters()), n=1)
            
            # Update target network
            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - option['soft_tau']) + param.data * option['soft_tau'])

            for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - option['soft_tau']) + param.data * option['soft_tau'])

            # Update theta <- theta*
            for ast_param, param in zip(ast_value_net.parameters(), value_net.parameters()):
                param.data.copy_(ast_param.data)
                
            for ast_param, param in zip(ast_policy_net.parameters(), policy_net.parameters()):
                param.data.copy_(ast_param.data)
        
        epoch_pl.append(np.mean(all_task_training_pl))
        epoch_vl.append(np.mean(all_task_training_vl))
    
    
    # ===========================================================================
    print("Policy\tstart:", epoch_pl[0], "\tend:", epoch_pl[-1], "\tmax:", np.max(epoch_pl), "\tmin:", np.min(epoch_pl), "\tmean:", np.mean(epoch_pl))
    print("Value\tstart:", epoch_vl[0], "\tend:", epoch_vl[-1], "\tmax:", np.max(epoch_vl), "\tmin:", np.min(epoch_vl), "\tmean:", np.mean(epoch_vl))

    json.dump(
        {
            "meta": option,
            "result": {
                "policy_loss": {
                    "max": np.max(epoch_pl), 
                    "min": np.min(epoch_pl), 
                    "mean": np.mean(epoch_pl),
                    "all": epoch_pl
                },
                "value_loss": {
                    "max": np.max(epoch_vl), 
                    "min": np.min(epoch_vl), 
                    "mean": np.mean(epoch_vl),
                    "all": epoch_vl
                }
            }
        },
        open("./meta_results/meta_run{}.json".format(len(os.listdir("./meta_results/"))), "w")
    )
    
    save_meta_path = os.path.join(option['path'], "meta_models")
    if not Path(save_meta_path).exists():
        os.makedirs(save_meta_path)
        
    policy_net.zero_grad()
    value_net.zero_grad()
    
    torch.save(policy_net.state_dict(), os.path.join(save_meta_path, "MetaPolicyNet.pt"))
    torch.save(value_net.state_dict(), os.path.join(save_meta_path, "MetaValueNet.pt"))
    torch.save(target_policy_net.state_dict(), os.path.join(save_meta_path, "MetaTargetPolicyNet.pt"))
    torch.save(target_value_net.state_dict(), os.path.join(save_meta_path, "MetaTargetValueNet.pt"))