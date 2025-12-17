import secrets
import argparse
import wandb
import torch
import torch.optim as optim
import numpy as np
import random
import os
from mpe2 import simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3

from utils import load_hyperparameters, save_stats, plot, save_metadata
from q_network import QNetwork
from training import Trainer
from agent import IQLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--env",
        type=str,
        default="simple_adversary_v3",
        choices=[
            "simple_adversary_v3",
            "simple_speaker_listener_v4",
            "simple_spread_v3",
        ],
    )
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")

    seed = secrets.randbelow(2**32)

    # Use parallel_env() for parallel multi-agent environment
    env = {
        "simple_adversary_v3": simple_adversary_v3.parallel_env(),
        "simple_world_comm_v3": simple_speaker_listener_v4.parallel_env(),
        "simple_crypto_v3": simple_spread_v3.parallel_env(),
    }[args.env]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs, info = env.reset(seed=seed)

    agents = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for agent_name in env.agents:
        obs_dim = env.observation_space(agent_name).shape[0]
        act_dim = env.action_space(agent_name).n

        q_net = QNetwork(obs_dim, act_dim)

        agent = IQLAgent(
            q_net=q_net,
            obs_dim=obs_dim,
            act_dim=act_dim,
            optimizer=None,  # Will set later
            scheduler=None,  # Will set later
            gamma=config["gamma"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            target_update_freq=config["target_update_freq"],
            device=device,
        )
        optimizer = optim.Adam(
            params=agent.q_net.parameters(), lr=config["learning_rate"]
        )
        agent.optimizer = optimizer

        total_training_steps = config["num_episodes"] * config["max_steps_per_episode"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_training_steps, eta_min=config["eta_min"]
        )

        agent.scheduler = scheduler
        agents[agent_name] = agent

    if args.train:
        os.makedirs("models", exist_ok=True)
        save_metadata(
            model=q_net,
            config=config,
            seed=seed,
            filepath=f"models/iql_{args.env}_seed_{seed}_metadata.json",
        )

        wandb.init(project="iql", config=config, name=f"{args.env}_seed_{seed}")
        trainer = Trainer(
            env=env,
            agents=agents,
            num_episodes=config["num_episodes"],
            max_steps_per_episode=config["max_steps_per_episode"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config["epsilon_decay"],
            device=device,
            checkpoint_interval=config["checkpoint_interval"],
            checkpoint_dir="checkpoints",
            seed=seed,
        )
        train_stats = trainer.train()
        os.makedirs("stats/training", exist_ok=True)
        save_stats(train_stats, f"stats/training/iql_{args.env}_seed_{seed}_stats.json")
        os.makedirs("plots", exist_ok=True)
        plot(train_stats, f"plots/iql_{args.env}_seed_{seed}_plot.png")

        wandb.finish()


if __name__ == "__main__":
    main()
