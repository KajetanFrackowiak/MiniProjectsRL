import random
import torch

from q_network import QNetwork


class QMIXAgent:
    """QMIX Agent: independent Q-network with shared mixing network.

    This agent maintains its own Q-network. The replay buffer and mixer network
    are managed at the trainer level for joint optimization across all agents.
    """

    def __init__(
        self,
        agent_id,
        q_net,
        obs_dim,
        act_dim,
        optimizer,
        scheduler,
        gamma,
        device="cuda",
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gamma = gamma

        # Agent's own Q-network and target Q-network
        self.q_net = q_net.to(device)
        self.target_q_net = QNetwork(obs_dim, act_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.device = device
        self.steps = 0

    def act(self, obs, eps):
        """Epsilon-greedy action selection."""
        if random.random() < eps:
            return random.randint(0, self.act_dim - 1)
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs)
        return q_values.argmax(dim=1).item()

    def update_target_network(self):
        """Hard update: copy online network to target network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def get_lr(self):
        """Get current learning rate."""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        return 0.0
