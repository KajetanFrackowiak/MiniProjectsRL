import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDDPG, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


class CriticDDPG(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticDDPG, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticTD3, self).__init__()
        # Critic 1
        self.fc1_1 = nn.Linear(state_dim, 400)
        self.fc2_1 = nn.Linear(400 + action_dim, 300)
        self.fc3_1 = nn.Linear(300, 1)

        # Critic 2
        self.fc1_2 = nn.Linear(state_dim, 400)
        self.fc2_2 = nn.Linear(400 + action_dim, 300)
        self.fc3_2 = nn.Linear(300, 1)

    def forward(self, state, action):
        # Critic 1 forward
        x1 = F.relu(self.fc1_1(state))
        x1 = torch.cat([x1, action], dim=1)
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)

        # Critic 2 forward
        x2 = F.relu(self.fc1_2(state))
        x2 = torch.cat([x2, action], dim=1)
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc3_2(x2)

        return x1, x2


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(ActorSAC, self).__init__()
        self.max_action = max_action

        # Shared layers
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(300, action_dim)

        # Constrain log_std to reasonable values
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # rsample() for reparameterization trick

        # Apply tanh and scale
        action = torch.tanh(x_t) * self.max_action

        # Calculate log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.max_action * (1 - action.pow(2) / (self.max_action**2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class CriticSAC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticSAC, self).__init__()
        # Critic 1
        self.fc1_1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2_1 = nn.Linear(400, 300)
        self.fc3_1 = nn.Linear(300, 1)

        # Critic 2
        self.fc1_2 = nn.Linear(state_dim + action_dim, 400)
        self.fc2_2 = nn.Linear(400, 300)
        self.fc3_2 = nn.Linear(300, 1)

    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)

        # Critic 1 forward
        x1 = F.relu(self.fc1_1(sa))
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)

        # Critic 2 forward
        x2 = F.relu(self.fc1_2(sa))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc3_2(x2)

        return x1, x2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1_1(sa))
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)
        return x1
