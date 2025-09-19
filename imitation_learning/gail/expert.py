import torch


class Expert:
    def __init__(self, episode, device="cpu"):
        self.actions = episode.actions
        self.step_idx = 0
        self.total_steps = len(self.actions)
        self.device = device

    def __call__(self, state=None):
        if self.step_idx >= self.total_steps:
            action = self.actions[-1]
        else:
            action = self.actions[self.step_idx]
            self.step_idx += 1
        return torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )  # [action_dim,] -> [1, action_dim]

    def reset(self):
        self.step_idx = 0
