import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = torch.zeros((capacity, *obs_shape), dtype=np.float32)
        # We need it in TD error
        self.next_obs_buf = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        
    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = torch.tensor(obs, device=self.obs_buf.device)
        self.next_obs_buf[self.ptr] = torch.tensor(next_obs, device=self.next_obs_buf.device)
        self.act_buf[self.ptr] = torch.tensor(act, device=self.act_buf.device)
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        # torch.randint expects tuple as a size
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs_buf.device)
        obs = self.obs_buf[idx]
        next_obs = self.next_obs_buf[idx]
        act = self.act_buf[idx]
        rew = self.rew_buf[idx]
        done = self.done_buf[idx]
        return obs, act, rew, next_obs, done