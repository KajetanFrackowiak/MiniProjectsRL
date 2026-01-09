import numpy as np
import torch


class EpisodeReplayBuffer:
    """Centralized episode replay buffer for QMIX.

    Stores complete episode trajectories with joint observations, actions,
    and global team rewards. Samples batches of episodes for training.
    """

    def __init__(
        self,
        capacity,
        n_agents,
        obs_dims,
        n_actions,
        max_episode_len,
        state_dim,
        device="cpu",
    ):
        """
        Args:
            capacity: Maximum number of episodes to store
            n_agents: Number of agents
            obs_dims: Dict or list of observation dimensions per agent
            n_actions: Number of actions per agent
            max_episode_len: Maximum timesteps per episode
            state_dim: Dimension of global state
            device: torch device
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.max_episode_len = max_episode_len
        self.device = device

        # Convert obs_dims to list if dict
        if isinstance(obs_dims, dict):
            self.obs_dims = [obs_dims[k] for k in sorted(obs_dims.keys())]
        else:
            self.obs_dims = obs_dims

        # Buffers: [capacity, max_episode_len, n_agents, feature_dim]
        self.obs_buffer = [
            torch.zeros(
                (capacity, max_episode_len, dim), dtype=torch.float32, device=device
            )
            for dim in self.obs_dims
        ]
        self.actions_buffer = torch.zeros(
            (capacity, max_episode_len, n_agents), dtype=torch.long, device=device
        )
        self.rewards_buffer = torch.zeros(
            (capacity, max_episode_len), dtype=torch.float32, device=device
        )
        self.dones_buffer = torch.zeros(
            (capacity, max_episode_len), dtype=torch.float32, device=device
        )
        self.states_buffer = torch.zeros(
            (capacity, max_episode_len, state_dim), dtype=torch.float32, device=device
        )
        self.episode_lengths = torch.zeros(capacity, dtype=torch.long, device=device)

        self.ptr = 0
        self.size = 0

    def add_episode(self, obs_list, actions_list, rewards, dones, states):
        """
        Add a complete episode to the buffer.

        Args:
            obs_list: List of observations per agent, each [episode_len, obs_dim]
            actions_list: List of actions per agent, each [episode_len]
            rewards: Global team rewards [episode_len]
            dones: Done flags [episode_len]
            states: Global states [episode_len, state_dim]
        """
        episode_len = len(rewards)

        # Store observations for each agent
        for agent_idx, obs in enumerate(obs_list):
            obs_array = np.array(obs, dtype=np.float32)
            self.obs_buffer[agent_idx][self.ptr, :episode_len] = torch.tensor(
                obs_array, dtype=torch.float32, device=self.device
            )

        # Store actions [episode_len, n_agents]
        actions_array = np.array(actions_list).T  # Transpose to [episode_len, n_agents]
        self.actions_buffer[self.ptr, :episode_len] = torch.tensor(
            actions_array, dtype=torch.long, device=self.device
        )

        # Store rewards and dones
        self.rewards_buffer[self.ptr, :episode_len] = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        self.dones_buffer[self.ptr, :episode_len] = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        # Store states
        states_array = np.array(states, dtype=np.float32)
        self.states_buffer[self.ptr, :episode_len] = torch.tensor(
            states_array, dtype=torch.float32, device=self.device
        )

        self.episode_lengths[self.ptr] = episode_len

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of episodes.

        Returns:
            obs: List of [batch_size, max_len, obs_dim] per agent
            actions: [batch_size, max_len, n_agents]
            rewards: [batch_size, max_len]
            dones: [batch_size, max_len]
            states: [batch_size, max_len, state_dim]
            lengths: [batch_size]
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        obs = [buf[indices] for buf in self.obs_buffer]
        actions = self.actions_buffer[indices]
        rewards = self.rewards_buffer[indices]
        dones = self.dones_buffer[indices]
        states = self.states_buffer[indices]
        lengths = self.episode_lengths[indices]

        return obs, actions, rewards, dones, states, lengths

    def __len__(self):
        return self.size
