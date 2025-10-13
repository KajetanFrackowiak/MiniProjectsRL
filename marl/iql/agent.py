import random
import torch
import torch.nn.functional as F

from q_network import QNetwork
from utils import ReplayBuffer


class IQLAgent:
    def __init__(
        self,
        q_net,
        obs_dim,
        act_dim,
        optimizer,
        scheduler,
        gamma,
        buffer_size=100_000,
        batch_size=64,
        tau=0.005,  # Polyak averaging coefficient
        device="cuda",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gamma = gamma
        self.tau = tau
        self.q_net = q_net.to(device)
        self.target_q_net = QNetwork(obs_dim, act_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size, obs_shape=(obs_dim,), device=device
        )
        self.batch_size = batch_size
        self.device = device
        self.steps = 0

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randint(0, self.act_dim - 1)
        obs = torch.tensor(obs, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs)
        return q_values.argmax().item()

    def store(self, obs, act, reward, next_obs, done):
        self.replay_buffer.add(obs, act, reward, next_obs, done)

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return
        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.batch_size)
        q = self.q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_q_net(next_obs).max(dim=1)[0]
            target = rew + self.gamma * (1 - done) * next_q  # TD target

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Polyak averaging (soft update) - update target network gradually
        with torch.no_grad():
            for target_param, param in zip(
                self.target_q_net.parameters(), self.q_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self.steps += 1
        return loss.item()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
