import torch
import torch.nn as nn
from utils import conv2d_size_out


class A3C(nn.Module):
    def __init__(self, input_dim, num_actions, noisy_nets=False):
        super(A3C, self).__init__()
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
        

        # batch_first=True allows the input to be of shape (batch_size, seq_length, input_dim)
        self.lstm = nn.LSTM(linear_input_size, 512, batch_first=True)
        self.fc = nn.Linear(512, 512)
        self.actor = nn.Linear(512, num_actions)  # (policy head)
        self.critic = nn.Linear(512, 1)  # (value head)

    def forward(self, x, hx=None, cx=None):
        # hx is the hidden state and cx is the cell state for LSTM
        # x: (batch_size, seq_length, input_dim, height, width)
        batch, seq_len, c, h, w = x.size()
        x = x.view(batch * seq_len, c, h, w)
        x = self.conv(x)  # (batch_size * seq_len, 64, convh, convw)
        x = x.view(batch, seq_len, -1)
        if hx is None or cx is None:
            lstm_out, (hx, cx) = self.lstm(x)
        else:
            lstm_out, (hx, cx) = self.lstm(x, (hx, cx))
        # lstm_out (batch_size, seq_len, 512)
        x = torch.relu(self.fc(lstm_out))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value, (hx, cx)

