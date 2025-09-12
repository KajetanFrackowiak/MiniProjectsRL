import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        def conv2d_size_out(size, kernel_size, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.fc(x)
        return Categorical(logits=logits)

class BitDiscretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Discretize to bits (0 or 1)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output

def discretize_bits(x):
    return BitDiscretizer.apply(x)

class WorldModelStochastic(nn.Module):
    def __init__(self, input_dim, num_actions, num_bits=32, input_height=84, input_width=84):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.enc_ln = nn.LayerNorm([64, 7, 7])

        # Action embedding
        self.action_emb = nn.Linear(num_actions, 64)

        # Inference network for posterior q(z|x, a, x')
        self.infer_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 64 + input_dim * input_height * input_width, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.infer_out = nn.Linear(128, num_bits)

        # LSTM prior for bits
        self.lstm_prior = nn.LSTM(input_size=num_bits, hidden_size=num_bits, num_layers=1, batch_first=True)

        # Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + num_bits, 64 * 7 * 7),
            nn.LayerNorm(64 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.dec3 = nn.ConvTranspose2d(32, 1*256, kernel_size=8, stride=4)

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(64 * 7 * 7 + num_bits + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.num_bits = num_bits
        self.input_dim = input_dim
        self.input_height = input_height
        self.input_width = input_width

    def encode(self, obs):
        x1 = F.relu(self.enc1(obs))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x3 = self.enc_ln(x3)
        return x1, x2, x3

    def infer_posterior(self, x3, a_emb, next_frame):
        # Flatten and concatenate with action embedding and next frame
        x_flat = x3.view(x3.size(0), -1)
        next_flat = next_frame.view(next_frame.size(0), -1)
        inp = torch.cat([x_flat, a_emb, next_flat], dim=1)
        h = self.infer_fc(inp)
        logits = self.infer_out(h)
        probs = torch.sigmoid(logits)
        # Use deterministic bits during evaluation, stochastic during training
        if self.training:
            # Sample bits (add controlled noise for stochasticity)
            noise = torch.rand_like(probs) - 0.5
            bits = discretize_bits(probs + 0.1 * noise)  # Reduced noise
        else:
            # Deterministic inference
            bits = discretize_bits(probs)
        return bits, probs

    def prior_lstm(self, prev_bits, hidden=None):
        # prev_bits: (B, T, num_bits)
        out, hidden = self.lstm_prior(prev_bits, hidden)
        return out, hidden

    def decode(self, x3, bits, a_emb, x1, x2):
        # Concatenate bottleneck with bits
        x_flat = x3.view(x3.size(0), -1)
        dec_in = torch.cat([x_flat, bits], dim=1)
        d = self.dec_fc(dec_in)
        d = d.view(-1, 64, 7, 7) 
        a_emb_64 = a_emb.view(a_emb.size(0), a_emb.size(1), 1, 1)
        a_emb_32 = a_emb[:, :32].contiguous().view(a_emb.size(0), 32, 1, 1)  # [B, 32, 1, 1]
        # Attention
        d = d * a_emb_64
        d = F.relu(self.dec1(d) + x2)
        d = d * a_emb_64
        d = F.relu(self.dec2(d) + x1)
        d = d * a_emb_32
        d = self.dec3(d)
        B, C256, H, W = d.shape
        next_frame_logits = d.view(B, 1, 256, H, W)
        return next_frame_logits

    def forward(self, obs, action, next_frame=None, mode='train', prev_bits=None, hidden=None):
        # obs: (B, input_dim, H, W)
        # action: (B, num_actions) one-hot
        # next_frame: (B, input_dim, H, W) (only for training)
        # mode: 'train' or 'inference'
        x1, x2, x3 = self.encode(obs)
        a_emb = self.action_emb(action)

        if mode == 'train':
            # Posterior inference network
            bits, bit_probs = self.infer_posterior(x3, a_emb, next_frame)
        else:
            # LSTM prior: auto-regressive bit prediction
            # prev_bits: (B, T, num_bits) where T is number of bits generated so far
            if prev_bits is None:
                prev_bits = torch.zeros(obs.size(0), 1, self.num_bits, device=obs.device)
            lstm_out, hidden = self.prior_lstm(prev_bits, hidden)
            # Use deterministic inference for consistency
            bit_probs = torch.sigmoid(lstm_out[:, -1, :])
            bits = discretize_bits(bit_probs)

        # Decoder
        next_frame_logits = self.decode(x3, bits, a_emb, x1, x2)

        # Reward prediction
        x_flat = x3.view(x3.size(0), -1)
        reward = self.reward_head(torch.cat([x_flat, bits, a_emb], dim=1))

        return next_frame_logits, reward, bits, bit_probs

# Usage:
# During training: model(obs, action, next_frame, mode='train')
# During inference: model(obs, action, mode='inference')