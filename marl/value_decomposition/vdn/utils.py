import os
import yaml
import json
import numpy as np
from datetime import datetime
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_hyperparameters(file_path):
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {file_path} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    return config


def save_params(data, file_name, dir_name):
    os.makedirs(dir_name, exist_ok=True)

    data_serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data_serializable[key] = value.tolist()
        else:
            data_serializable[key] = value

    file_path = os.path.join(dir_name, file_name)
    with open(file_path, "w") as f:
        json.dump(data_serializable, f, indent=4)
    print(f"Saved to {file_path}")


def load_params(file_name, dir_name):
    file_path = os.path.join(dir_name, file_name)
    with open(file_path, "r") as f:
        params = json.load(f)
    return params


def collect_results_across_seeds(results_dir, env_name):
    seed_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith("seed_")])

    all_rewards = []
    all_steps = []
    all_losses = []

    for seed_dir in seed_dirs:
        stats_file = os.path.join(
            results_dir, seed_dir, "training_stats", "train_stats.json"
        )
        if os.path.exists(stats_file):
            stats = load_params(
                "train_stats.json",
                os.path.join(results_dir, seed_dir, "training_stats"),
            )
            if stats.get("env_name") == env_name:
                all_rewards.append(stats.get("reward", []))
                all_steps.append(stats.get("steps", []))
                all_losses.append(stats.get("loss", []))
            else:
                print(
                    f"Warning: {seed_dir} has different env_name: {stats.get('env_name')}"
                )
        else:
            print(f"Warning: {stats_file} not found")

    train_stats = {
        "reward": np.array(all_rewards),
        "steps": np.array(all_steps),
        "loss": np.array(all_losses),
    }

    return train_stats


def plot_all_envs(results_dir, env_names):
    titles = ["reward", "steps", "loss"]
    y_labels = ["Cumulative Reward", "Steps per Episode", "Loss"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(len(env_names), 3, figsize=(15, 4 * len(env_names)))

    if len(env_names) == 1:
        axes = axes.reshape(1, -1)

    for env_idx, env_name in enumerate(env_names):
        train_stats = collect_results_across_seeds(results_dir, env_name)

        for metric_idx, (title, y_label, color) in enumerate(
            zip(titles, y_labels, colors)
        ):
            ax = axes[env_idx, metric_idx]
            data = train_stats[title]

            if data.size == 0:
                print(f"Warning: No data for {env_name} - {title}")
                continue

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            num_episodes = np.arange(len(mean))

            ax.plot(num_episodes, mean, color=color, linewidth=2, label="Mean")

            ax.fill_between(
                num_episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
                label="±1 Std Dev",
            )

            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(
                f"{env_name} - {title.capitalize()}", fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

    plt.tight_layout()
    os.makedirs("plots")
    plt.savefig("plots/results_all_envs.png", dpi=300, bbox_inches="tight")
    print("Saved plot to plots/results_all_envs.png")
    plt.show()


def save_metadata(
    agent, config, seed, env_name, file_path="metadata.json", dir_name="metadata"
):
    os.makedirs(dir_name, exist_ok=True)

    trainable_params = 0
    non_trainable_params = 0

    for q_net in agent.q_networks:
        trainable_params += np.sum(np.prod(v.shape) for v in q_net.trainable_variables)
        non_trainable_params += np.sum(
            np.prod(v.shape) for v in q_net.non_trainable_variables
        )

    trainable_params += np.sum(
        np.prod(v.shape) for v in agent.mixing_network.trainable_variables
    )
    non_trainable_params += np.sum(
        np.prod(v.shape) for v in agent.mixing_network.non_trainable_variables
    )

    total_params = trainable_params + non_trainable_params

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(seed),
        "environment": env_name,
        "num_agents": int(agent.num_agents),
        "model_parameters": {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(non_trainable_params),
            "num_q_networks": int(agent.num_agents),
            "obs_dims": [int(d) for d in agent.obs_dims],
            "act_dim": int(agent.act_dim),
            "state_dim": int(agent.state_dim),
        },
        "config": config,
    }

    file_path_full = os.path.join(dir_name, file_path)
    with open(file_path_full, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"Metadata saved to {file_path_full}")
    print(
        f"Total parameters: {total_params}, Trainable: {trainable_params}, Non-trainable: {non_trainable_params}"
    )


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, is_multi_agent=False):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buf = (
            np.empty(capacity, dtype=object)
            if is_multi_agent
            else np.zeros(capacity, dtype=np.int32)
        )
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = tf.convert_to_tensor(self.obs_buf[idx], dtype=tf.float32)
        next_obs = tf.convert_to_tensor(self.next_obs_buf[idx], dtype=tf.float32)
        acts = self.act_buf[idx]
        rews = tf.convert_to_tensor(self.rew_buf[idx], dtype=tf.float32)
        dones = tf.convert_to_tensor(self.done_buf[idx], dtype=tf.float32)
        return obs, next_obs, acts, rews, dones
