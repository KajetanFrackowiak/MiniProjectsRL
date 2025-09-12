import torch
from models import C51Network
from utils import ReplayBuffer


class C51Agent:
    def __init__(
        self,
        stacked_frames,
        act_dim,
        num_atoms=51,
        Vmin=-10,
        Vmax=10,
        lr=1e-4,
        buffer_capacity=100000,
        device=None
     ):
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.z = torch.linspace(Vmin, Vmax, num_atoms, device=device)  # (num_atoms,)
        self.target_network = C51Network(stacked_frames, act_dim, num_atoms).to(device)
        self.policy_network = C51Network(stacked_frames, act_dim, num_atoms).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.device = device

    def get_action(self, state):
        logits = self.policy_network(state)  # (batch_size, num_actions, num_atoms)
        # Probability distribution (over atoms) for each possible action
        probs = torch.softmax(logits, dim=-1)  # (batch_size, num_actions, num_atoms)
        q_values = torch.sum(probs * self.z, dim=2)
        action = torch.argmax(q_values, dim=1)
        return action

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        logits = self.policy_network(state)  # (batch_size, num_actions, num_atoms)
        probs = torch.softmax(logits, dim=2) # (batch_size, num_actions, num_atoms)
        dist = probs[range(batch_size), action]  # (batch_size, num_atoms)

        with torch.no_grad():
            next_logits = self.target_network(next_state)  # (batch_size, num_actions, num_atoms)
            next_probs = torch.softmax(next_logits, dim=2) # (batch_size, num_actions, num_atoms)
            next_q = torch.sum(next_probs * self.z, dim=2)  # (batch_size, num_actions)
            next_action = torch.argmax(next_q, dim=1) # (batch_size,)
            # Select, for every sample, the probability distribution (over atoms) for its own best next action
            next_dist = next_probs[range(batch_size), next_action] # (batch_size, num_atoms)
            target_dist = self._project_distribution(reward, done, next_dist) # (batch_size, num_atoms)

        loss = -torch.sum(target_dist * torch.log(dist + 1e-10), dim=1).mean()
        loss.backward()
        self.optimizer.step()
        

    def _project_distribution(self, reward, done, next_dist, gamma=0.99):
        # reward.shape = (batch_size,)
        batch_size = reward.shape[0]
        target_dist = torch.zeros((batch_size, self.num_atoms), device=next_dist.device)
        # delta_z is the width of each atom
        delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)
        for b in range(batch_size):
            for j in range(self.num_atoms):
                # Bellman update for each atom
                Tz = reward[b] + (1 - done[b]) * gamma * self.z[j]
                Tz = Tz.clamp(self.Vmin, self.Vmax)
                # bj is the index of the atom in the target distribution
                bj = (Tz - self.Vmin) / delta_z
                # l is the lower index of the atom
                l = torch.floor(bj).long()
                # u is the upper index of the atom
                u = torch.ceil(bj).long()
                # Distribute probability to nearest atoms
                if l == u:
                    target_dist[b, l] += next_dist[b, j]
                else:
                    # Distribute the probability mass to the lower atom
                    target_dist[b, l] += next_dist[b, j] * (u.float() - bj)
                    # Distribute the probability mass to the upper atom
                    target_dist[b, u] += next_dist[b, j] * (bj - l.float())
        return target_dist
