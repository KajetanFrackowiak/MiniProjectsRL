import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import secrets
import yaml
import os
from tqdm import tqdm
from agent import AgentDDPG, AgentTD3, AgentSAC


def load_hyperparameters():
    with open("hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(agent, checkpoint_path):
    average_returns = []
    if os.path.exists(checkpoint_path):
        average_returns, episode_rewards, starting_episode, seed = agent.load(
            checkpoint_path
        )
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
    return average_returns, episode_rewards, starting_episode, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="1: Hopper-v4: Walker2d-v4: HalfCheetah-v4: Ant-v4: Humanoid-v4",
    )
    parser.add_argument("--seed", type=int, default=secrets.randbelow(2**32))
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--method", type=int, choices=[1, 2, 3], help="1: DDPG 2: TD3 3: SAC"
    )
    args = parser.parse_args()

    env_names = {
        1: "Hopper-v4",
        2: "Walker2d-v4",
        3: "HalfCheetah-v4",
        4: "Ant-v4",
        5: "Humanoid-v4",
    }

    # Map environment numbers to environment step config keys
    env_step_keys = {
        1: "hopper",
        2: "walker2d",
        3: "halfcheetah",
        4: "ant",
        5: "humanoid",
    }
    ENV_NAME = env_names[args.env]
    env = gym.make(ENV_NAME)

    config = load_hyperparameters()

    # Get environment-specific training steps and calculate episodes
    env_key = env_step_keys[args.env]
    total_timesteps = config["environment_steps"][env_key]
    MAX_TIMESTEPS = config["general"]["max_timesteps"]
    # Don't pre-calculate episodes - train until we reach total_timesteps
    EVAL_FREQ = config["general"]["eval_freq"]

    print(f"Environment: {ENV_NAME}")
    print(f"Training for {total_timesteps:,} timesteps")
    print(f"Max timesteps per episode: {MAX_TIMESTEPS}")
    print(f"Evaluation frequency: every {EVAL_FREQ} episodes")

    # Algorithm-specific parameters
    if args.method == 1:
        LEARNING_RATE = config["ddpg"]["learning_rate"]
        GAMMA = config["ddpg"]["gamma"]
        TAU = config["ddpg"]["tau"]
        BUFFER_SIZE = config["ddpg"]["buffer_size"]
        BATCH_SIZE = config["ddpg"]["batch_size"]
        OU_MU = config["ddpg"]["ou_mu"]
        OU_THETA = config["ddpg"]["ou_theta"]
        OU_SIGMA = config["ddpg"]["ou_sigma"]
        WARMUP_STEPS = config["ddpg"]["warmup_steps"]

    elif args.method == 2:
        LEARNING_RATE = config["td3"]["learning_rate"]
        BUFFER_SIZE = config["td3"]["buffer_size"]
        BATCH_SIZE = config["td3"]["batch_size"]
        GAMMA = config["td3"]["gamma"]
        TAU = config["td3"]["tau"]
        POLICY_NOISE = config["td3"]["policy_noise"]
        NOISE_CLIP = config["td3"]["noise_clip"]
        POLICY_FREQ = config["td3"]["policy_freq"]
        EXPLORATION_NOISE = config["td3"]["exploration_noise"]
        WARMUP_STEPS = config["td3"]["warmup_steps"]

    elif args.method == 3:
        LEARNING_RATE = config["sac"]["learning_rate"]
        BUFFER_SIZE = config["sac"]["buffer_size"]
        BATCH_SIZE = config["sac"]["batch_size"]
        GAMMA = config["sac"]["gamma"]
        TAU = config["sac"]["tau"]
        ALPHA = config["sac"]["alpha"]
        AUTOMATIC_ENTROPY_TUNING = config["sac"]["automatic_entropy_tuning"]
        WARMUP_STEPS = config["sac"]["warmup_steps"]

    if args.method == 1:
        agent = AgentDDPG(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            ou_mu=OU_MU,
            ou_theta=OU_THETA,
            ou_sigma=OU_SIGMA,
        )
    elif args.method == 2:
        agent = AgentTD3(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ,
        )
        # Set exploration noise from config
        agent.noise_std = EXPLORATION_NOISE
    elif args.method == 3:
        agent = AgentSAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            automatic_entropy_tuning=AUTOMATIC_ENTROPY_TUNING,
        )

    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        average_returns, episode_rewards, starting_episode, seed = load_checkpoint(
            agent, checkpoint_path
        )
        print(
            f"Loaded checkpoint from {checkpoint_path} with average returns: {average_returns} and starting episode: {starting_episode}"
        )
    else:
        average_returns = []
        episode_rewards = []
        starting_episode = 0
        seed = args.seed

    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    methods_dict = {1: "DDPG", 2: "TD3", 3: "SAC"}

    # Training parameters - align learning start with warmup end
    start_learning_steps = WARMUP_STEPS  # Start learning after warmup period
    total_steps = 0

    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Learning starts after: {start_learning_steps} steps")

    episode = starting_episode
    # Create progress bar for total training steps
    pbar = tqdm(total=total_timesteps, desc="Training Progress", unit="steps")
    pbar.update(total_steps)  # Update to current position if resuming

    while total_steps < total_timesteps:
        state, _ = env.reset()
        episode_reward = 0
        for t in range(MAX_TIMESTEPS):
            # Random actions during warmup period for better exploration
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            # Update progress bar
            pbar.update(1)

            # Start learning only after sufficient experience
            if total_steps > start_learning_steps and len(agent.memory) > BATCH_SIZE:
                agent.learn()  # Check if we've reached total training steps
            if total_steps >= total_timesteps:
                break

            if done:
                break

        episode_rewards.append(episode_reward)
        average_return = np.mean(episode_rewards[-100:])
        average_returns.append(average_return)

        if (episode + 1) % EVAL_FREQ == 0:
            learning_status = (
                "Learning" if total_steps > start_learning_steps else "Warmup"
            )
            progress_pct = (total_steps / total_timesteps) * 100
            # Use tqdm.write to avoid conflicts with progress bar
            pbar.write(
                f"Episode {episode + 1}, Average Return: {average_return:.2f}, Steps: {total_steps}/{total_timesteps} ({progress_pct:.1f}%), Status: {learning_status}"
            )

            os.makedirs("models", exist_ok=True)

            if args.method == 3:
                checkpoint_path = f"models/checkpoint_episode_{episode + 1}_method_{methods_dict[args.method]}_env_{ENV_NAME}_aet_{AUTOMATIC_ENTROPY_TUNING}.pth"
            else:
                checkpoint_path = f"models/checkpoint_episode_{episode + 1}_method_{methods_dict[args.method]}_env_{ENV_NAME}.pth"
            agent.save(
                checkpoint_path=checkpoint_path,
                average_returns=average_returns,
                episode_rewards=episode_rewards,
                starting_episode=starting_episode,
                seed=seed,
            )

        episode += 1

    pbar.close()  # Close progress bar when training is complete
    env.close()


if __name__ == "__main__":
    main()
    print("Training completed.")
