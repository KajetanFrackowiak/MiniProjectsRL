import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=True):
        super().__init__()
        self.discrete = discrete

        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 100)

        if self.discrete:
            self.action_head = nn.Linear(100, action_dim)
        else:
            self.mean_head = nn.Linear(100, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        if self.discrete:
            logits = self.action_head(x)
            return logits
        else:
            mean = self.mean_head(x)
            std = torch.exp(self.log_std)
            return mean, std


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)

        x = self.fc1(x)
        x = F.tanh(x)

        x = self.fc2(x)
        x = F.tanh(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        return x
