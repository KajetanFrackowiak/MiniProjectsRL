import tensorflow as tf
import keras
from mpe2 import simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3
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
        help="simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3",
    )
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")

    if args.env == "simple_spread_v3":
        env = simple_spread_v3.parallel_env(max_cycles=config["max_cycles"])
    elif args.env == "simple_speaker_listener_v4":
        env = simple_speaker_listener_v4.parallel_env(max_cycles=config["max_cycles"])
    elif args.env == "simple_adversary_v3":
        env = simple_adversary_v3.parallel_env(max_cycles=config["max_cycles"])
    else:
        raise ValueError(
            f"Unknown environment: {args.env}. Choose from: simple_spread_v3, simple_speaker_listener_v4, simple_adversary_v3"
        )

    max_steps_per_episode = config["max_cycles"]
    total_train_steps = (
        config["num_episodes"] * max_steps_per_episode // config["scheduler_frequency"]
    )

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        config["learning_rate"], decay_steps=total_train_steps, alpha=config["alpha"]
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    seed = secrets.randbelow(2**32)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    agent_ids = (
        env.agents if hasattr(env, "agents") else list(env.observation_spaces.keys())
    )
    num_agents = len(agent_ids)

    obs_dims = [env.observation_space(agent_id).shape[0] for agent_id in agent_ids]
    act_dim = env.action_space(agent_ids[0]).n

    test_obs_dict, _ = env.reset()
    test_observations = [np.array(test_obs_dict[agent_id]) for agent_id in agent_ids]
    test_state = np.concatenate(test_observations)
    state_dim = test_state.shape[0]

    print(f"Environment: {args.env}")
    print(
        f"Agents: {num_agents}, Obs dims: {obs_dims}, Act dim: {act_dim}, State dim: {state_dim}"
    )

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

    wandb.init(project="VDN", name=f"{args.env}_seed_{seed}", mode="disabled")
    print("Starting training...")

    trainer = Trainer(
        env=env,
        agent=agent,
        optimizer=optimizer,
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        ckpt_manager=ckpt_manager,
        checkpoint_freq=config["checkpoint_freq"],
    )

    train_stats = trainer.train(num_episodes=config["num_episodes"])

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
