from keras import layers, Model
import tensorflow as tf
from typing import Tuple


class QNetwork(Model):
    """
    Individual Q-network for each agent in VDN.

    Processes agent observations and outputs Q-values for each action.
    Architecture: obs_dim -> 256 (ReLU) -> act_dim
    """

    def __init__(self, obs_dim: int, act_dim: int):
        """
        Args:
            obs_dim: Dimension of agent observations
            act_dim: Number of actions available
        """
        super(QNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Use Xavier/Glorot initialization for better training stability
        self.dense1 = layers.Dense(
            256,
            activation="relu",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="hidden",
        )
        self.dense2 = layers.Dense(
            act_dim,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="output",
        )

    def build(self, input_shape: Tuple) -> None:
        """Build network layers."""
        super(QNetwork, self).build(input_shape)
        self.dense1.build(input_shape)
        self.dense2.build((input_shape[0], 256))
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        x = self.dense1(inputs)
        return self.dense2(x)


class MixingNetwork(Model):
    """
    Mixing network for Value Decomposition Networks (VDN).

    Combines individual agent Q-values into a global Q-value using
    non-linear mixing controlled by the state.

    Q_total = sum(w_i * Q_i) + b
    where w_i and b are functions of the global state.
    """

    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int = 32):
        """
        Args:
            num_agents: Number of agents
            state_dim: Dimension of global state
            hidden_dim: Hidden dimension for w_net and b_net
        """
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.w_net = tf.keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu", name="w_hidden"),
                layers.Dense(num_agents, activation="relu", name="w_output"),
            ],
            name="w_network",
        )

        self.b_net = tf.keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu", name="b_hidden"),
                layers.Dense(1, name="b_output"),
            ],
            name="b_network",
        )

    def build(self, input_shape: Tuple) -> None:
        """Build network layers."""
        super(MixingNetwork, self).build(input_shape)

        # Build w_net and b_net with state input shape
        state_shape = (input_shape[0], self.state_dim)
        self.w_net.build(state_shape)
        self.b_net.build(state_shape)

        self.built = True

    def call(self, individual_q_values: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        """
        Args:
            individual_q_values: (batch_size, num_agents) Q-values from each agent
            state: (batch_size, state_dim) Global state

        Returns:
            (batch_size,) Global mixed Q-values
        """
        w = self.w_net(state)  # (batch_size, num_agents)
        w = tf.nn.relu(w)

        q_total = tf.reduce_sum(w * individual_q_values, axis=1)  # (batch_size,)

        b = self.b_net(state)  # (batch_size, 1)
        q_total = q_total + tf.squeeze(b, axis=1)

        return q_total
