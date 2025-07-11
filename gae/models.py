import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class LinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.linear(obs)
        # Clamp log_std to prevent numerical instability
        log_std_clamped = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(log_std_clamped)

        if torch.isnan(std).any() or torch.isnan(mean).any():
            print("Nan detected")
            print("Mean:", mean)
            print("Std:", std)
            print("Log std:", log_std_clamped)
            exit()

        return Normal(mean, std)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 20), nn.Tanh(), nn.Linear(20, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
        )
        self.mean = nn.Linear(25, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.body(obs)
        mean = self.mean(x)
        # Clamp log_std to prevent numerical instability
        log_std_clamped = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(log_std_clamped)
        return Normal(mean, std)


class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class CNNDiscretePolicy(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        def conv2d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Check for NaN inputs
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in CNN input! Shape: {x.shape}")
            print(f"NaN locations: {torch.isnan(x).sum()}")

        x = self.conv(x)

        # Check for NaN after conv layers
        if torch.isnan(x).any():
            print(f"Warning: NaN detected after conv layers! Shape: {x.shape}")

        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        # Check for NaN in logits
        if torch.isnan(logits).any():
            print(f"Warning: NaN detected in logits! Shape: {logits.shape}")
            print(
                f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}"
            )
            # Clamp logits to prevent NaN
            logits = torch.clamp(logits, min=-20, max=20)

        return Categorical(logits=logits)


class CNNValueFunction(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        def conv2d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x):
        # Check for NaN inputs
        if torch.isnan(x).any():
            print(
                f"Warning: NaN detected in CNN Value function input! Shape: {x.shape}"
            )

        x = self.conv(x)

        # Check for NaN after conv layers
        if torch.isnan(x).any():
            print(
                f"Warning: NaN detected in CNN Value function after conv layers! Shape: {x.shape}"
            )

        x = x.view(x.size(0), -1)
        value = self.fc(x)

        # Check for NaN in values
        if torch.isnan(value).any():
            print(
                f"Warning: NaN detected in CNN Value function output! Shape: {value.shape}"
            )
            value = torch.clamp(value, min=-1000, max=1000)

        return value.squeeze(-1)  # Returns shape (batch,)
