import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        self.act_dim = act_dim

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x


class QMixNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_embed=64):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed

        # Hypernetwork for generating weights of first layer
        # Output should be n_agents * embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed_dim),
        )

        # Hypernetwork for generating bias of first layer
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for generating weights of second layer
        # Output should be embed_dim * 1
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )

        # Hypernetwork for generating bias of second layer (scalar)
        # V(s) in the QMIX paper
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global state [batch_size, state_dim]

        Returns:
            q_tot: Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        # Reshape agent_qs to [batch_size, 1, n_agents] for batch matrix multiplication with weights
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # First layer
        # Generate weights and ensure they are non-negative (for monotonicity)
        w1 = torch.abs(self.hyper_w1(states))  # [batch_size, n_agents * embed_dim]
        # We need to reshape weights to perform batch matrix multiplication
        w1 = w1.view(
            batch_size, self.n_agents, self.embed_dim
        )  # [batch_size, n_agents, embed_dim]

        b1 = self.hyper_b1(states)  # [batch_size, embed_dim]
        b1 = b1.view(batch_size, 1, self.embed_dim)  # [batch_size, 1, embed_dim]

        # Matrix multiplication: [batch_size, 1, n_agents] x [batch_size, n_agents, embed_dim]
        hidden = torch.bmm(agent_qs, w1) + b1  # [batch_size, 1, embed_dim]
        hidden = F.elu(hidden)

        # Second layer
        # Generate weights and ensure they are non-negative
        w2 = torch.abs(self.hyper_w2(states))  # [batch_size, embed_dim]
        w2 = w2.view(batch_size, self.embed_dim, 1)  # [batch_size, embed_dim, 1]

        b2 = self.hyper_b2(states)  # [batch_size, 1]
        b2 = b2.view(batch_size, 1, 1)  # [batch_size, 1, 1]

        # Matrix multiplication: [batch_size, 1, embed_dim] x [batch_size, embed_dim, 1]
        q_tot = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
        q_tot = q_tot.view(batch_size, 1)  # [batch_size, 1]

        return q_tot
