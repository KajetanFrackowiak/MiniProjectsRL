import tensorflow as tf
import numpy as np

from utils import ReplayBuffer
from q_network import QNetwork, MixingNetwork


class VDNAgent:
    """Value Decomposition Network agent for multi-agent RL."""

    def __init__(
        self,
        num_agents,
        obs_dims,
        act_dim,
        state_dim,
        optimizer,
        gamma,
        buffer_size,
        batch_size,
        tau,
    ):
        self.num_agents = num_agents
        self.obs_dims = (
            [obs_dims] * num_agents if isinstance(obs_dims, int) else obs_dims
        )
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q_networks = [
            QNetwork(self.obs_dims[i], act_dim) for i in range(num_agents)
        ]
        self.target_q_networks = [
            QNetwork(self.obs_dims[i], act_dim) for i in range(num_agents)
        ]

        self.mixing_network = MixingNetwork(num_agents, state_dim, hidden_dim=32)
        self.target_mixing_network = MixingNetwork(num_agents, state_dim, hidden_dim=32)

        # Explicitly build networks
        for i, q_net in enumerate(self.q_networks):
            q_net.build((None, self.obs_dims[i]))
        for i, q_net in enumerate(self.target_q_networks):
            q_net.build((None, self.obs_dims[i]))

        self.mixing_network.build((None, num_agents))
        self.target_mixing_network.build((None, num_agents))

        # Single unified replay buffer that stores all agent observations
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            (state_dim,),
            is_multi_agent=True,
            num_agents=num_agents,
            obs_dims=obs_dims,
        )

        self.steps = 0
        self.update_counter = 0

        # Running reward statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def actions(self, observations, state, epsilon, avail_actions=None):
        actions = []
        for i, (obs, q_net) in enumerate(zip(observations, self.q_networks)):
            if np.random.rand() < epsilon:
                if avail_actions is not None:
                    available = np.nonzero(avail_actions[i])[0]
                    actions.append(np.random.choice(available))
                else:
                    actions.append(np.random.randint(self.act_dim))
            else:
                obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
                q_values = q_net(tf.expand_dims(obs_tensor, axis=0))[0]
                if avail_actions is not None:
                    q_values = q_values - (1 - avail_actions[i]) * 1e9
                actions.append(tf.argmax(q_values).numpy())
        return actions

    def store(
        self, observations, state, actions, reward, next_observations, next_state, done
    ):
        # Update reward statistics for normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        # Running variance using Welford's algorithm
        if self.reward_count == 1:
            self.reward_std = 1.0
        else:
            self.reward_std = np.sqrt(
                ((self.reward_count - 1) * self.reward_std**2 + delta * delta2)
                / self.reward_count
            )

        # Normalize reward
        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        # Store all data in a unified buffer structure
        self.replay_buffer.add(
            state=state,
            next_state=next_state,
            actions=tuple(actions),
            reward=normalized_reward,
            done=done,
            observations=observations,
            next_observations=next_observations,
        )

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None

        # Sample from unified buffer
        batch = self.replay_buffer.sample(self.batch_size)
        state = batch["state"]
        next_state = batch["next_state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        observations = batch["observations"]
        next_observations = batch["next_observations"]

        actions = np.array([list(a) for a in actions], dtype=np.int32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Current Q-values for actions taken
            individual_qs = []
            for i, (obs, q_net) in enumerate(zip(observations, self.q_networks)):
                q_values = q_net(obs)
                batch_indices = tf.range(tf.shape(actions)[0])
                indices = tf.stack([batch_indices, actions[:, i]], axis=1)
                q = tf.gather_nd(q_values, indices)
                individual_qs.append(q)

            individual_qs = tf.stack(individual_qs, axis=1)
            q_total = self.mixing_network(individual_qs, state)

            # Target Q-values
            target_individual_qs = []
            for i, (next_obs, target_q_net) in enumerate(
                zip(next_observations, self.target_q_networks)
            ):
                next_q_values = target_q_net(next_obs)
                next_q_max = tf.reduce_max(next_q_values, axis=1)
                target_individual_qs.append(next_q_max)

            target_individual_qs = tf.stack(target_individual_qs, axis=1)
            target_q_total = self.target_mixing_network(
                target_individual_qs, next_state
            )
            target_q_total = rewards + self.gamma * tf.stop_gradient(target_q_total) * (
                1 - dones
            )

            # Huber loss instead of MSE for robustness to outliers
            td_error = q_total - target_q_total
            # Clip TD error to prevent exploding values
            td_error = tf.clip_by_value(td_error, -10.0, 10.0)
            loss = tf.reduce_mean(tf.square(td_error))

        all_trainable_vars = [
            v for net in self.q_networks for v in net.trainable_variables
        ] + self.mixing_network.trainable_variables
        grads = tape.gradient(loss, all_trainable_vars)

        # Gradient clipping to prevent exploding gradients
        grads, global_norm = tf.clip_by_global_norm(grads, 10.0)

        self.optimizer.apply_gradients(zip(grads, all_trainable_vars))

        self.update_counter += 1
        # Update target networks less frequently for stability
        if self.update_counter % 1 == 0:
            for target_q, q in zip(self.target_q_networks, self.q_networks):
                for target_var, var in zip(
                    target_q.trainable_variables, q.trainable_variables
                ):
                    target_var.assign(self.tau * var + (1 - self.tau) * target_var)

            for target_var, var in zip(
                self.target_mixing_network.trainable_variables,
                self.mixing_network.trainable_variables,
            ):
                target_var.assign(self.tau * var + (1 - self.tau) * target_var)

        return loss.numpy()

    def get_lr(self):
        return self.optimizer.learning_rate.numpy()
