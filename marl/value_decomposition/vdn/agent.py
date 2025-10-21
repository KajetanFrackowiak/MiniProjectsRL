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

        self.replay_buffer = ReplayBuffer(
            buffer_size, (state_dim,), is_multi_agent=True
        )
        self.agent_obs_buffer = [
            ReplayBuffer(buffer_size, (self.obs_dims[i],), is_multi_agent=False)
            for i in range(num_agents)
        ]

        self.steps = 0

    def actions(self, observations, state, epsilon):
        actions = []
        for i, (obs, q_net) in enumerate(zip(observations, self.q_networks)):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(self.act_dim))
            else:
                obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
                q_values = q_net(tf.expand_dims(obs_tensor, axis=0))
                actions.append(tf.argmax(q_values[0]).numpy())
        return actions

    def store(
        self, observations, state, actions, reward, next_observations, next_state, done
    ):
        self.replay_buffer.add(state, next_state, tuple(actions), reward, done)
        for i, (obs, next_obs) in enumerate(zip(observations, next_observations)):
            self.agent_obs_buffer[i].add(obs, next_obs, actions[i], reward, done)

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None

        state, next_state, actions, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )

        actions = np.array([list(a) for a in actions], dtype=np.int32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        agent_obs_batch = []
        agent_next_obs_batch = []
        for i in range(self.num_agents):
            obs, next_obs, _, _, _ = self.agent_obs_buffer[i].sample(self.batch_size)
            agent_obs_batch.append(obs)
            agent_next_obs_batch.append(next_obs)

        with tf.GradientTape() as tape:
            individual_qs = []
            for i, (obs, q_net) in enumerate(zip(agent_obs_batch, self.q_networks)):
                q_values = q_net(obs)
                batch_indices = tf.range(tf.shape(actions)[0])
                indices = tf.stack([batch_indices, actions[:, i]], axis=1)
                q = tf.gather_nd(q_values, indices)
                individual_qs.append(q)

            individual_qs = tf.stack(individual_qs, axis=1)

            q_total = self.mixing_network(individual_qs, state)

            target_individual_qs = []
            for i, (next_obs, target_q_net) in enumerate(
                zip(agent_next_obs_batch, self.target_q_networks)
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

            loss = tf.reduce_mean((q_total - target_q_total) ** 2)

        all_trainable_vars = [
            v for net in self.q_networks for v in net.trainable_variables
        ] + self.mixing_network.trainable_variables
        grads = tape.gradient(loss, all_trainable_vars)
        self.optimizer.apply_gradients(zip(grads, all_trainable_vars))

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
