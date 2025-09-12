import torch.nn as nn


class C51Network(nn.Module):
    def __init__(self, stacked_frames, act_dim, num_atoms=51):
        super().__init__()
        self.conv = nn.Sequential(
            # H_out = floor(84 + 2 * 0 - 8) / 4 + 1 = 20
            # W_out = floor(84 + 2 * 0 - 8) / 4 + 1 = 20
            nn.Conv2d(
                stacked_frames, 32, kernel_size=8, stride=4
            ),  # (batch_size, 32, 20, 20) = (batch_size, channels, height, width)
            nn.ReLU(),
            # H_out = floor(20 + 2 * 0 - 4) / 2 + 1 = 9
            # W_out = floor(20 + 2 * 0 - 4) / 2 + 1 = 9
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2
            ),  # (batch_size, 64, 9, 9) = (batch_size, channels, height, width)
            nn.ReLU(),
            # H_out = floor(9 + 2 * 0 - 3) / 1 + 1 = 7
            # W_out = floor(9 + 2 * 0 - 3) / 1 + 1 = 7
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1
            ),  # (batch_size, 64, 7, 7) = (batch_size, channels, height, width)
            nn.ReLU(),
        )

        def conv2d_size_out(size, kernel_size=1, stride=1):
            return (size - kernel_size) // stride + 1

        # We need to calculate the output size after the convolutional layers
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        
        linear_input_size = convw * convh * 64  # (7 * 7 * 64)

        self.num_atoms = num_atoms
        self.act_dim = act_dim
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim * num_atoms),
        )

    def forward(self, x):
        x = self.conv(x)
        # (batch_size, channels, height, width) -> (batch_size, channels * height * width)
        x = x.view(x.size(0), -1) # Flatten to be able to put into FC layer
        # (batch_size, channels * height * width) -> (batch_size, act_dim * num_atoms)
        x = self.fc(x)
        # (batch_size, act_dim * num_atoms) -> (batch_size, act_dim, num_atoms)
        return x.view(x.size(0), self.act_dim, self.num_atoms)
