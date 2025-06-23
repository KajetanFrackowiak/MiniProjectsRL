import torch
import torch.nn as nn
from torch.distributions import Normal

class LinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.linear(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 100), nn.Tanh(),
            nn.Linear(100, 50), nn.Tanh(),
            nn.Linear(50, 25), nn.Tanh()
        )
        self.mean = nn.Linear(25, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.body(obs)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 100), nn.Tanh(),
            nn.Linear(100, 50), nn.Tanh(),
            nn.Linear(50, 25), nn.Tanh(),
            nn.Linear(25, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
