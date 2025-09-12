import torch
import torch.optim as optim
import numpy as np
import os
from models import ActorDDPG, CriticDDPG, CriticTD3, ActorSAC, CriticSAC
from utils import ReplayBuffer


class BaseAgent:
    """Base class for RL agents with common functionality."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        learning_rate,
        gamma,
        tau,
        buffer_size,
        batch_size,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(buffer_size)

    def _soft_update(self, target, source, tau):
        """Soft update target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def _prepare_batch(self, batch_size):
        """Common batch preparation for learning."""
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        return states, actions, rewards, next_states, dones


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state = self.state + dx
        return self.state


class AgentDDPG(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        ou_mu=0.0,
        ou_theta=0.15,
        ou_sigma=0.2,
    ):
        super().__init__(
            state_dim,
            action_dim,
            max_action,
            learning_rate,
            gamma,
            tau,
            buffer_size,
            batch_size,
        )

        # Networks
        self.actor = ActorDDPG(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorDDPG(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = CriticDDPG(state_dim, action_dim).to(self.device)
        self.critic_target = CriticDDPG(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Noise for exploration
        self.ou_noise = OUNoise(action_dim, ou_mu, ou_theta, ou_sigma)
        self.initial_ou_sigma = ou_sigma  # Store initial noise level
        self.noise_decay = 0.9999  # Decay factor for exploration noise

    def select_action(self, state, add_noise=True):
        # (state_dim,) -> (1, state_dim) because actor expects dim: [1, state_dim]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            action += self.ou_noise.sample()
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def learn(self):
        batch_data = self._prepare_batch(self.batch_size)
        if batch_data is None:
            return None

        states, actions, rewards, next_states, dones = batch_data

        # Update Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            # Bootstrapped estimate of the Q-value
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (self.gamma * target_q * ~dones)

        current_q = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks, instead of replacing them directly like in DQN
        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.actor_target, self.actor, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def load(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_target.load_state_dict(checkpoint["actor_target"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])

            # Load optimizer states if available
            if "optimizer_state_dict" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "critic_optimizer_state_dict" in checkpoint:
                self.critic_optimizer.load_state_dict(
                    checkpoint["critic_optimizer_state_dict"]
                )

            average_returns = checkpoint.get("average_returns", [])
            episode_rewards = checkpoint.get("episode_rewards", [])
            starting_episode = checkpoint.get("starting_episode", 0)
            seed = checkpoint.get("seed", None)
            print(f"Model loaded from {checkpoint_path}")
            return average_returns, episode_rewards, starting_episode, seed
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return [], [], 0, None

    def save(
        self, checkpoint_path, average_returns, episode_rewards, starting_episode, seed
    ):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "average_returns": average_returns,
                "episode_rewards": episode_rewards,
                "starting_episode": starting_episode,
                "seed": seed,
            },
            checkpoint_path,
        )
        print(f"Model saved to {checkpoint_path}")


class AgentTD3(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        super().__init__(
            state_dim,
            action_dim,
            max_action,
            learning_rate,
            gamma,
            tau,
            buffer_size,
            batch_size,
        )

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # Networks
        self.actor = ActorDDPG(state_dim, action_dim, max_action).to(
            self.device
        )  # TD3 uses same actor as DDPG
        self.actor_target = ActorDDPG(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = CriticTD3(state_dim, action_dim).to(self.device)
        self.critic_target = CriticTD3(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Training step counter
        self.total_steps = 0
        self.noise_std = 0.1

    def select_action(self, state, add_noise=True):
        # (state_dim,) -> (1, state_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            action += np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def learn(self):
        batch_data = self._prepare_batch(self.batch_size)
        if batch_data is None:
            return None

        self.total_steps += 1
        states, actions, rewards, next_states, dones = batch_data

        # Update Critics
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            target_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (self.gamma * target_q * ~dones)

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = torch.nn.functional.mse_loss(
            current_q1, target_q
        ) + torch.nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_steps % self.policy_freq == 0:
            # Update Actor
            actor_loss = -self.critic(states, self.actor(states))[
                0
            ].mean()  # Use first critic for actor update

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.critic_target, self.critic, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss else None,
        }

    def load(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_target.load_state_dict(checkpoint["actor_target"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])
            self.total_steps = checkpoint["total_steps"]
            self.noise_std = checkpoint["noise_std"]
            self.actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            average_returns = checkpoint.get("average_returns", [])
            episode_rewards = checkpoint.get("episode_rewards", [])
            starting_episode = checkpoint.get("starting_episode", 0)
            seed = checkpoint.get("seed", None)
            print(f"Model loaded from {checkpoint_path}")
            return average_returns, episode_rewards, starting_episode, seed
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return [], [], 0, None

    def save(
        self, checkpoint_path, average_returns, episode_rewards, starting_episode, seed
    ):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "total_steps": self.total_steps,
                "noise_std": self.noise_std,
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "average_returns": average_returns,
                "episode_rewards": episode_rewards,
                "starting_episode": starting_episode,
                "seed": seed,
            },
            checkpoint_path,
        )
        print(f"Model saved to {checkpoint_path}")


class AgentSAC(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        alpha=0.2,
        automatic_entropy_tuning=True,
    ):
        super().__init__(
            state_dim,
            action_dim,
            max_action,
            learning_rate,
            gamma,
            tau,
            buffer_size,
            batch_size,
        )

        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Networks
        self.actor = ActorSAC(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticSAC(state_dim, action_dim).to(self.device)
        self.critic_target = CriticSAC(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor([action_dim]).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def select_action(self, state, deterministic=False):
        # (state_dim,) -> (1, state_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
        else:
            action, _ = self.actor.sample(state)

        return action.cpu().data.numpy().flatten()

    def learn(self):
        batch_data = self._prepare_batch(self.batch_size)
        if batch_data is None:
            return None

        states, actions, rewards, next_states, dones = batch_data

        # Update Critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (self.gamma * target_q * ~dones)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(
            current_q1, target_q
        ) + torch.nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if alpha_loss else None,
            "alpha": self.alpha.item()
            if isinstance(self.alpha, torch.Tensor)
            else self.alpha,
        }

    def load(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])

            # Load optimizer states
            self.actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )

            # Load entropy tuning parameters
            if self.automatic_entropy_tuning:
                self.log_alpha = checkpoint["log_alpha"].to(self.device)
                self.alpha = self.log_alpha.exp()
                if "alpha_optimizer_state_dict" in checkpoint:
                    self.alpha_optimizer.load_state_dict(
                        checkpoint["alpha_optimizer_state_dict"]
                    )

            average_returns = checkpoint.get("average_returns", [])
            episode_rewards = checkpoint.get("episode_rewards", [])
            starting_episode = checkpoint.get("starting_episode", 0)
            seed = checkpoint.get("seed", None)

            print(f"Model loaded from {checkpoint_path}")
            return average_returns, episode_rewards, starting_episode, seed
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return [], [], 0, None

    def save(
        self, checkpoint_path, average_returns, episode_rewards, starting_episode, seed
    ):
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "average_returns": average_returns,
            "episode_rewards": episode_rewards,
            "starting_episode": starting_episode,
            "seed": seed,
        }

        # Save entropy tuning parameters if using automatic entropy tuning
        if self.automatic_entropy_tuning:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha"] = self.alpha
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
