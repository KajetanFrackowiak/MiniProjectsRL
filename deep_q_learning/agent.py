import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from models import DQN, DuelingDQN
from utils import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    def __init__(
        self,
        input_dims,
        num_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
    ):
        # (num_stacked_frames, height, width)
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_dims[0], num_actions).to(self.device)
        self.target_net = DQN(input_dims[0], num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        # Epsilon-greedy exploration
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay_steps
        )

        if random.random() > epsilon:
            with torch.no_grad():
                # state.shape = (num_stacked_frames, height, width)
                # state_tensor.shape = (batch_size, num_stacked_frames, height, width)
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                # q_values.shape = (batch_size, num_actions (values for each item))
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples to learn

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # policy_net.shape = (batch_size, num_actions) (input for gather, but gather is interested only num_actions)
        # actions.shape = (batch_size,), we need actions.unsqueeze(1) = (batch_size, 1) for gather
        # Gather q_values along num_actions
        current_q_values = (
            self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_q_values_target = self.target_net(next_states).max(1)[0]
        # If next state is terminal, its Q-value is 0
        next_q_values_target[dones.bool()] = 0.0

        expected_q_values = rewards + (self.gamma * next_q_values_target)

        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()
        print(f"Model loaded from. Resuming with steps_done: {self.steps_done}")


class DoubleDQNAgent(DQNAgent):
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # The input tensor: self.policy_net(states) has two dimensions: (batch_size, num_actions)
        # But actions has one dimension: (batch_size,), actions.unsqueeze(1) -> (batch_size, 1) lets .gather()
        # pick exactly one action index per batch element along dimension 1. It outputs (batch_size, 1) tensor. But we need (batch_size,) tensor in downstream code like loss computation.
        # So we use .squeeze(1) to remove the second dimension.
        current_q_values = (
            self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            # Instead of using target_net directly, we use policy_net to select actions
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # And then use target_net to get Q-values for those actions
            next_q_values_target = (
                self.target_net(next_states)
                .gather(dim=1, index=next_actions)
                .squeeze(1)
            )
        next_q_values_target[dones.bool()] = 0.0

        expected_q_values = rewards + (self.gamma * next_q_values_target)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()


class PrioritizedReplayAgent(DoubleDQNAgent):
    def __init__(
        self,
        input_dims,
        num_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        mode="proportional",
    ):
        super().__init__(
            input_dims,
            num_actions,
            learning_rate,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
            buffer_size,
            batch_size,
            target_update_freq,
        )
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=alpha, mode=mode)

    def store_transition(self, state, action, reward, next_state, done):
        # Assign max priority for new transition
        max_priority = self.memory.max_priority() if len(self.memory) > 0 else 1.0
        self.memory.push(state, action, reward, next_state, done, max_priority)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        # Anneal beta for importance-sampling weights
        beta = self.beta_by_frame(self.steps_done)
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, beta=beta)
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Instead of using torch.tensor(weights, dtype=torch.float32), we use torch.clone().detach() to ensure that the weights are not affected by the gradients.
        weights = weights.clone().detach().to(self.device)
        current_q_values = (
            self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values_target = (
                self.target_net(next_states)
                .gather(dim=1, index=next_actions)
                .squeeze(1)
            )
        next_q_values_target[dones.bool()] = 0.0

        expected_q_values = rewards + (self.gamma * next_q_values_target)

        # Compute TD error
        td_errors = current_q_values - expected_q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in buffer
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def beta_by_frame(self, frame_idx):
        # Linear annealing of beta from beta_start to 1.0 over beta_frames
        return min(
            1.0,
            self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames,
        )


class PrioritizedDuelingAgent(PrioritizedReplayAgent):
    def __init__(
        self,
        input_dims,
        num_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        mode="proportional",
    ):
        super().__init__(
            input_dims,
            num_actions,
            learning_rate,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
            buffer_size,
            batch_size,
            target_update_freq,
            alpha,
            beta_start,
            beta_frames,
            mode,
        )
        self.policy_net = DuelingDQN(input_dims[0], num_actions).to(self.device)
        self.target_net = DuelingDQN(input_dims[0], num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        beta = self.beta_by_frame(self.steps_done)
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, beta=beta)
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        current_q_values = (
            self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values_target = (
                self.target_net(next_states)
                .gather(dim=1, index=next_actions)
                .squeeze(1)
            )
            next_q_values_target[dones.bool()] = 0.0

        expected_q_values = rewards + (self.gamma * next_q_values_target)

        # Compute TD error
        td_errors = current_q_values - expected_q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()
