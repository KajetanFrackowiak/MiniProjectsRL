import secrets
import argparse
import wandb
import torch
import torch.optim as optim
import numpy as np
import random
import os
from smac.env import StarCraft2Env
from mpe2 import simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3

from utils import load_hyperparameters, save_stats, plot, save_metadata
from q_network import QNetwork, QMixNetwork
from training import Trainer
from agent import QMIXAgent


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
            "3m",
            "8m",
            "3s5z",
        ],
    )
    args = parser.parse_args()

    # Auto-select hyperparameters based on environment type
    is_smac = args.env in ["3m", "8m", "3s5z"]
    config_file = "hyperparameters_smac.yaml" if is_smac else "hyperparameters_mpe.yaml"
    config = load_hyperparameters(config_file)

    # Environment-specific overrides for MPE
    if args.env == "simple_adversary_v3":
        config["epsilon_decay_steps"] = 15000
        print(
            f"Using config: {config_file} (adversary overrides: epsilon_decay_steps=15000)"
        )
    elif args.env == "simple_speaker_listener_v4":
        config["epsilon_decay_steps"] = 30000
        config["learning_rate"] = 0.0003
        print(
            f"Using config: {config_file} (speaker_listener overrides: epsilon_decay_steps=30000, lr=0.0003)"
        )
    # Environment-specific overrides for SMAC
    elif args.env == "8m":
        config["buffer_size"] = 8000
        config["num_episodes"] = 15000
        print(
            f"Using config: {config_file} (8m overrides: buffer_size=8000, num_episodes=15000)"
        )
    elif args.env == "3s5z":
        config["buffer_size"] = 10000
        config["num_episodes"] = 20000
        config["mixer_embed_dim"] = 128
        config["mixer_hypernet_embed"] = 256
        print(
            f"Using config: {config_file} (3s5z overrides: buffer_size=10000, num_episodes=20000, mixer=128/256)"
        )
    else:
        print(f"Using config: {config_file}")

    # Reproducible seed
    seed = secrets.randbelow(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Use parallel_env() for parallel multi-agent environment
    max_cycles = config.get("max_steps_per_episode", 100)
    env = {
        "simple_adversary_v3": simple_adversary_v3.parallel_env(max_cycles=max_cycles),
        "simple_speaker_listener_v4": simple_speaker_listener_v4.parallel_env(
            max_cycles=max_cycles
        ),
        "simple_spread_v3": simple_spread_v3.parallel_env(max_cycles=max_cycles),
        "3m": StarCraft2Env(map_name="3m"),
        "8m": StarCraft2Env(map_name="8m"),
        "3s5z": StarCraft2Env(map_name="3s5z"),
    }[args.env]

    is_smac = args.env in ["3m", "8m", "3s5z"]

    if is_smac:
        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
        obs_dim = env_info["obs_shape"]
        state_dim = env_info["state_shape"]
        act_dim = env_info["n_actions"]
        env.reset()
    else:
        obs, info = env.reset(seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build agents with QMIX: each agent has own Q-net, shared mixer
    agents = {}
    q_nets = {}

    if is_smac:
        # SMAC: agents are indexed 0..n-1
        for agent_id in range(n_agents):
            agent_name = f"agent_{agent_id}"
            q_net = QNetwork(obs_dim, act_dim)
            q_nets[agent_name] = q_net

            agent = QMIXAgent(
                agent_id=agent_name,
                q_net=q_net,
                obs_dim=obs_dim,
                act_dim=act_dim,
                optimizer=None,
                scheduler=None,
                gamma=config["gamma"],
                device=device,
            )
            optimizer = optim.Adam(
                params=agent.q_net.parameters(), lr=config["learning_rate"]
            )
            agent.optimizer = optimizer

            total_training_steps = (
                config["num_episodes"] * config["max_steps_per_episode"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_training_steps, eta_min=config["eta_min"]
            )
            agent.scheduler = scheduler
            agents[agent_name] = agent
    else:
        # MPE: use env.agents list
        for agent_name in env.agents:
            obs_dim = env.observation_space(agent_name).shape[0]
            act_dim = env.action_space(agent_name).n

            q_net = QNetwork(obs_dim, act_dim)
            q_nets[agent_name] = q_net

            agent = QMIXAgent(
                agent_id=agent_name,
                q_net=q_net,
                obs_dim=obs_dim,
                act_dim=act_dim,
                optimizer=None,
                scheduler=None,
                gamma=config["gamma"],
                device=device,
            )
            optimizer = optim.Adam(
                params=agent.q_net.parameters(), lr=config["learning_rate"]
            )
            agent.optimizer = optimizer

            total_training_steps = (
                config["num_episodes"] * config["max_steps_per_episode"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_training_steps, eta_min=config["eta_min"]
            )
            agent.scheduler = scheduler
            agents[agent_name] = agent

    # Create shared QMIX mixer network
    if is_smac:
        # SMAC provides state_dim directly
        mixer_state_dim = state_dim
        mixer_n_agents = n_agents
    else:
        # MPE: concatenate all observations as global state
        mixer_n_agents = len(env.agents)
        mixer_state_dim = sum(env.observation_space(a).shape[0] for a in env.agents)
    mixer = QMixNetwork(
        n_agents=mixer_n_agents,
        state_dim=mixer_state_dim,
        embed_dim=config.get("mixer_embed_dim", 32),
        hypernet_embed=config.get("mixer_hypernet_embed", 64),
    ).to(device)

    # Store whether this is a SMAC environment for the trainer
    env.is_smac = is_smac

    mixer_optimizer = optim.Adam(params=mixer.parameters(), lr=config["learning_rate"])

    if args.train:
        os.makedirs("models", exist_ok=True)
        save_metadata(
            model=list(q_nets.values())[0],
            config=config,
            seed=seed,
            filepath=f"models/qmix_{args.env}_seed_{seed}_metadata.json",
        )

        wandb.init(
            project="qmix",
            config=config,
            name=f"{args.env}_seed_{seed}",
            mode="online",
        )

        trainer = Trainer(
            env=env,
            agents=agents,
            mixer=mixer,
            mixer_optimizer=mixer_optimizer,
            num_episodes=config["num_episodes"],
            max_steps_per_episode=config["max_steps_per_episode"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay_steps=config.get("epsilon_decay_steps"),
            batch_size=config["batch_size"],
            buffer_size=config["buffer_size"],
            train_interval=config.get("train_interval", 4),
            target_update_freq=config["target_update_freq"],
            device=device,
            checkpoint_interval=config["checkpoint_interval"],
            checkpoint_dir="checkpoints",
            eval_interval=config.get("eval_interval", 50),
            eval_episodes=config.get("eval_episodes", 5),
            eval_max_steps=config.get("eval_max_steps", 200),
            seed=seed,
        )
        train_stats = trainer.train()

        os.makedirs("stats/training", exist_ok=True)
        save_stats(
            train_stats, f"stats/training/qmix_{args.env}_seed_{seed}_stats.json"
        )
        os.makedirs("plots", exist_ok=True)
        plot(train_stats, f"plots/qmix_{args.env}_seed_{seed}_plot.png")

        wandb.finish()


if __name__ == "__main__":
    main()
