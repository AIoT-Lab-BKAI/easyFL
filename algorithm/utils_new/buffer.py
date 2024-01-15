import math
import random
import numpy as np
import os
from pathlib import Path
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        state, action, reward, next_state, done = map(self.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def stack(self, list):
        return list
    
    def __len__(self):
        return len(self.buffer)

    def save(self, path, name):
        if not Path(path).exists():
            os.makedirs(path)

        with open(os.path.join(path, f'{name}.pkl'), 'wb') as file:
            pickle.dump(self.buffer, file)
        return
    
    def load(self, path, name):
        if Path(path).exists():
            try:
                with open(os.path.join(path, name), 'rb') as file:
                    self.buffer += pickle.load(file)
                return True
            except:
                print(f"Failed to load buffer from {os.path.join(path, f'{name}.pkl')}")
        else:
            print(f"{path} does not exist!")
        return False
            

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        
    def get_last_record(self):
        if len(self.states) < 2:
            return None
        else:
            return self.states[-2], self.actions[-2], self.rewards[-1], self.states[-1]
    
    def act(self, s, a):
        # at state s, agent performs action a
        self.actions.append(a)
        self.states.append(s)

    def update(self, r):
        # update reward and next state s' after perform last action a in state s
        # self.states.append(next_s)
        self.rewards.append(r)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()