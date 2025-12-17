import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.act_dim = act_dim

        self.linear1 = nn.Linear(obs_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, act_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x
