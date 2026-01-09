import tensorflow as tf
import keras
from mpe2 import simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3
from smac.env import StarCraft2Env
import wandb
import argparse
import secrets
import numpy as np

from agent import VDNAgent
from training import Trainer
from utils import (
    load_hyperparameters,
    save_metadata,
    save_params,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="MPE: simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3"
        "SMAC: 3m, 8m, 3s5z",
    )
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")
    seed = secrets.randbelow(2**32)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args.env == "simple_spread_v3":
        env = simple_spread_v3.parallel_env(max_cycles=config["max_cycles"])
    elif args.env == "simple_speaker_listener_v4":
        env = simple_speaker_listener_v4.parallel_env(max_cycles=config["max_cycles"])
    elif args.env == "simple_adversary_v3":
        env = simple_adversary_v3.parallel_env(max_cycles=config["max_cycles"])
    elif args.env in ["3m", "8m", "3s5z"]:
        env = StarCraft2Env(map_name=args.env, seed=seed)
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

    if args.env in [
        "simple_spread_v3",
        "simple_speaker_listener_v4",
        "simple_adversary_v3",
    ]:
        env.reset(seed=seed)

    if args.env == "simple_spread_v3" or args.env == "simple_speaker_listener_v4":
        num_episodes = config["num_episode_simple_spread_and_speaker"]
    elif args.env == "simple_adversary_v3":
        num_episodes = config["num_episode_simple_adversary"]
    elif args.env == "3m":
        num_episodes = config["num_episode_3m"]
    elif args.env == "8m":
        num_episodes = config["num_episode_8m"]
    elif args.env == "3s5z":
        num_episodes = config["num_episode_3s5z"]

    if args.env in ["3m", "8m", "3s5z"]:
        env_info = env.get_env_info()
        num_agents = env_info["n_agents"]
        agent_ids = list(range(num_agents))
        obs_size = env.get_obs_size()
        obs_dims = [obs_size] * num_agents
        act_dim = env_info["n_actions"]
        state_dim = obs_size * num_agents
        episode_limit = env_info["episode_limit"]
        total_train_steps = num_episodes * episode_limit
    else:
        agent_ids = list(env.observation_spaces.keys())
        num_agents = len(agent_ids)
        obs_dims = [env.observation_space(agent_id).shape[0] for agent_id in agent_ids]
        act_dim = env.action_space(agent_ids[0]).n
        state_dim = sum(obs_dims)
        total_train_steps = num_episodes * config["max_cycles"]

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        config["learning_rate"], decay_steps=total_train_steps, alpha=config["alpha"]
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    agent = VDNAgent(
        num_agents=num_agents,
        obs_dims=obs_dims,
        act_dim=act_dim,
        state_dim=state_dim,
        optimizer=optimizer,
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        tau=config["tau"],
    )

    save_metadata(agent, config, seed, args.env)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        mixing_net=agent.mixing_network,
        q_net_0=agent.q_networks[0],
    )
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, directory="checkpoints", max_to_keep=5
    )
    status = checkpoint.restore(ckpt_manager.latest_checkpoint)
    if status:
        print("Checkpoint restored")
    else:
        print("Training from scratch")

    wandb.init(project="VDN", name=f"{args.env}_seed_{seed}", config=config)
    print("Starting training...")

    trainer = Trainer(
        env=env,
        agent=agent,
        seed=seed,
        optimizer=optimizer,
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        ckpt_manager=ckpt_manager,
        checkpoint_freq=config["checkpoint_freq"],
    )

    train_stats = trainer.train(num_episodes=num_episodes)

    train_stats = {
        "env_name": args.env,
        "reward": train_stats.get("reward", []),
        "steps": train_stats.get("steps", []),
        "loss": train_stats.get("loss", []),
    }
    save_params(
        train_stats, f"train_stats_env_{args.env}_seed_{seed}.json", "training_stats"
    )


if __name__ == "__main__":
    main()
