import torch
import numpy as np
import yaml
import json
import datetime
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import matplotlib.pyplot as plt


def load_hyperparameters(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_stats(stats, filepath):
    with open(filepath, "w") as f:
        json.dump(stats, f)


def load_stats(filepath):
    with open(filepath, "r") as f:
        stats = json.load(f)
    return stats


def save_metadata(model, config, seed, filepath):
    # p.numel() gives the number of elements in the tensor
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    total_params = trainable_params + non_trainable_params

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "model_parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
        },
        "config": config,
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {filepath}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")


def plot(stats, save_path):

    episodes = list(range(1, len(stats["ep_avg_rewards"]) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, stats["ep_avg_rewards"], label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Episode Average Reward over Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episodes, stats["ep_avg_losses"], label="Average Loss", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Episode Average Loss over Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = torch.zeros(
            (capacity, *obs_shape), dtype=torch.float32, device=device
        )
        self.next_obs_buf = torch.zeros(
            (capacity, *obs_shape), dtype=torch.float32, device=device
        )
        self.act_buf = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=device)

        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = torch.tensor(obs, device=self.obs_buf.device)
        self.next_obs_buf[self.ptr] = torch.tensor(
            next_obs, device=self.next_obs_buf.device
        )
        self.act_buf[self.ptr] = torch.tensor(act, device=self.act_buf.device)
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # torch.randint expects tuple as a size
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs_buf.device)
        obs = self.obs_buf[idx]
        next_obs = self.next_obs_buf[idx]
        act = self.act_buf[idx]
        rew = self.rew_buf[idx]
        done = self.done_buf[idx]
        return obs, act, rew, next_obs, done
