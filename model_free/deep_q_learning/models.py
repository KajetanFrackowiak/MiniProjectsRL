import torch
import torch.nn as nn
import torch.functional as F
import math


class NoisyLinearFactorized(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinearFactorized, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.register_buffer("weight_epsilon_i", torch.empty(in_features))
        self.register_buffer("weight_epsilon_j", torch.empty(out_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        def f(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        self.weight_epsilon_i = f(torch.randn(self.in_features))
        self.weight_epsilon_j = f(torch.randn(self.out_features))
        self.bias_epsilon = self.weight_epsilon_j.clone()

    def forward(self, x):
        if self.training:
            weight_epsilon = torch.outer(self.weight_epsilon_j, self.weight_epsilon_i)
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class NoisyLinearUnfactorized(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinearUnfactorized, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon = torch.randn(self.out_features, self.in_features)
        self.bias_epsilon = torch.randn(self.out_features)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


def conv2d_size_out(size, kernel_size=1, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions, noisy_nets=False):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.noisy_nets = noisy_nets

        linear_layer = NoisyLinearFactorized if noisy_nets else nn.Linear

        self.fc = nn.Sequential(
            linear_layer(linear_input_size, 512),
            nn.ReLU(),
            linear_layer(512, num_actions),
        )

    def forward(self, x):
        # (batch_size, input_dim, height, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def reset_noise(self):
        if self.noisy_nets:
            # Loop through all layers and submodules inside the model
            for module in self.modules():
                if isinstance(module, NoisyLinearFactorized):
                    module.reset_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, num_actions, noisy_nets=False):
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

        self.noisy_nets = noisy_nets
        linear_layer = NoisyLinearFactorized if noisy_nets else nn.Linear

        self.value_stream = nn.Sequential(
            linear_layer(linear_input_size, 512),
            nn.ReLU(),
            linear_layer(512, 1),  # Output one value per state
        )

        self.advantage_stream = nn.Sequential(
            linear_layer(linear_input_size, 512),
            nn.ReLU(),
            linear_layer(512, num_actions),  # Output one advantage per action
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)  # (batch_size, 1)
        advantage = self.advantage_stream(x)  # (batch_size, num_action)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_vals = q_vals * (1 / math.sqrt(2))
        return q_vals

    def reset_noise(self):
        if self.noisy_nets:
            for module in self.modules():
                if isinstance(module, NoisyLinearFactorized):
                    module.reset_noise()


class Rainbow(nn.Module):
    def __init__(self, input_dim, act_dim, noisy_nets=False, num_atoms=51):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # We need to calculate the output size after the convolutional layers
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.noisy_nets = noisy_nets

        linear_layer = NoisyLinearFactorized if noisy_nets else nn.Linear

        self.num_atoms = num_atoms
        self.act_dim = act_dim

        self.value_stream = nn.Sequential(
            linear_layer(linear_input_size, 512),
            nn.ReLU(),
            linear_layer(512, num_atoms),
        )

        self.advantage_stream = nn.Sequential(
            linear_layer(linear_input_size, 512),
            nn.ReLU(),
            linear_layer(512, act_dim * num_atoms),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        value = self.value_stream(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(-1, self.act_dim, self.num_atoms)

        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # Normalize across atoms
        q_dist = torch.softmax(q_atoms, dim=2)  # (batch_size, action_dim, num_atoms)
        return q_dist

    def reset_noise(self):
        if self.noisy_nets:
            for module in self.modules():
                if isinstance(module, NoisyLinearFactorized):
                    module.reset_noise()
