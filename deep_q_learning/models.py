import torch
import torch.nn as nn
import math

def conv2d_size_out(size, kernel_size=1, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # (batch_size, input_dim, height, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.state_value = nn.Sequential(
             nn.Linear(linear_input_size, 512),
             nn.ReLU(),
             nn.Linear(512, 1)  # Output one value per state
        )
        
        self.advantage = nn.Sequential(
             nn.Linear(linear_input_size, 512),
             nn.ReLU(),
             nn.Linear(512, num_actions)  # Output one advantage per action
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        state_value = self.state_value(x)  # (batch_size, 1)
        advantage = self.advantage(x)  # (batch_size, num_action)
        qvals = state_value + (advantage - advantage.mean(dim=1, keepdim=True))
        qvals = qvals * (1 / math.sqrt(2))
        return  qvals
