import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from models import DQN
from utils import ReplayBuffer

class DQNAgent:
    def __init__(self, input_dims, num_actions, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=100000,
                 buffer_size=100000, batch_size=32, target_update_freq=1000):
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
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-1. * self.steps_done / self.epsilon_decay_steps)

        if random.random() > epsilon:
            with torch.no_grad():
                # state.shape = (num_stacked_frames, height, width)
                # state_tensor.shape = (batch_size, num_stacked_frames, height, width)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)


        # policy_net.shape = (batch_size, num_actions) (input for gather, but gather is interested only num_actions)
        # actions.shape = (batch_size,), we need actions.unsqueeze(1) = (batch_size, 1) for gather
        # Gather q_values along num_actions
        current_q_values = self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

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
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done
        }, path)

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
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # The input tensor: self.policy_net(states) has two dimensions: (batch_size, num_actions)
        # But actions has one dimension: (batch_size,), actions.unsqueeze(1) -> (batch_size, 1) lets .gather()
        # pick exactly one action index per batch element along dimension 1. It outputs (batch_size, 1) tensor. But we need (batch_size,) tensor in downstream code like loss computation.
        # So we use .squeeze(1) to remove the second dimension.
        current_q_values = self.policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Instead of using target_net directly, we use policy_net to select actions
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # And then use target_net to get Q-values for those actions
            next_q_values_target = self.target_net(next_states).gather(dim=1, index=next_actions).squeeze(1)
        next_q_values_target[dones.bool()] = 0.0
        
        expected_q_values = rewards + (self.gamma * next_q_values_target)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()