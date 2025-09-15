import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 128)
        self.linear2 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.sigmoid(x)

        return x

