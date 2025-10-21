from keras import layers, Model
import tensorflow as tf


class QNetwork(Model):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(act_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class MixingNetwork(Model):
    def __init__(self, num_agents, state_dim, hidden_dim=32):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.w_net = tf.keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(num_agents, activation="relu"),
            ]
        )

        self.b_net = tf.keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(1),
            ]
        )

    def call(self, individual_q_values, state):
        w = self.w_net(state)
        w = tf.nn.relu(w)

        q_total = tf.reduce_sum(w * individual_q_values, axis=1)

        b = self.b_net(state)
        q_total = q_total + tf.squeeze(b, axis=1)

        return q_total
