import torch
import numpy as np
import os
import wandb
import argparse
import gymnasium as gym
import ale_py
import yaml
import secrets
from mujoco_env import MuJoCoCartPoleEnv
from models import (
    LinearPolicy,
    ValueFunction,
    MLPPolicy,
    MLPValueFunction,
    CNNDiscretePolicy,
    CNNValueFunction,
)
from gae import trpo_update, compute_gae, update_value_function, ppo_update
from utils import preprocess_frame, FrameStacker


def load_hyperparameters(env_id, algorithm_id):
    """Load hyperparameters from YAML config file"""
    try:
        with open("hyperparameters.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: hyperparameters.yaml file not found!")
        print(
            "Please make sure the hyperparameters.yaml file is in the current directory."
        )
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

    # Map environment IDs to names
    env_map = {
        1: "pong",
        2: "cartpole",
        3: "pendulum",
        4: "ant",
        5: "humanoid",
        6: "cartpole_own",
    }

    # Map algorithm IDs to names
    algo_map = {1: "trpo", 2: "ppo"}

    env_name = env_map[env_id]
    algo_name = algo_map[algorithm_id]

    try:
        return (
            config[algo_name][env_name],
            config["environment_settings"],
            config["general"],
        )
    except KeyError as e:
        print(f"Error: Missing configuration for {algo_name}-{env_name}: {e}")
        print("Please check your hyperparameters.yaml file.")
        exit(1)


def collect_episodes(args, env, policy, value_fn, target_timesteps, gamma, device):
    observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    episode_returns = []
    episode_return = 0
    episode_length = 0
    timesteps_collected = 0

    obs = env.reset()[0]

    while timesteps_collected < target_timesteps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Debug: Check for NaN in observations
        if torch.isnan(obs_tensor).any():
            print(f"NaN detected in observation at timestep {timesteps_collected}")
            print("Observation:", obs)
            print("Obs tensor:", obs_tensor)
            break

        dist = policy(obs_tensor)
        action = dist.sample()

        # Handle action format based on environment type
        if args.env == 2:  # CartPole (discrete)
            # For discrete environments using continuous policy, convert action to discrete
            # CartPole has 2 actions, so we use the first component and threshold it
            action_continuous = action.squeeze(0).cpu().numpy()
            if hasattr(action_continuous, "__len__") and len(action_continuous) > 0:
                action_for_env = 1 if action_continuous[0] > 0 else 0
            else:
                action_for_env = 1 if action_continuous > 0 else 0
            log_prob = dist.log_prob(action)
            # Ensure scalar log_prob
            while log_prob.dim() > 0:
                log_prob = log_prob.sum()
        elif args.env in [3, 6]:  # Pendulum and CartPole_own (continuous)
            # For Pendulum: action shape should be (batch_size, action_dim) = (1, 1)
            action_for_env = action.squeeze(0).cpu().numpy()
            # For 1D action spaces, get scalar log_prob
            log_prob = dist.log_prob(action.squeeze(0))
            # Ensure log_prob is a scalar
            while log_prob.dim() > 0:
                log_prob = log_prob.sum()
        elif args.env in [4, 5]:  # Ant, Humanoid (continuous)
            action_for_env = action.squeeze(0).cpu().numpy()
            log_prob = dist.log_prob(action.squeeze(0))
            # Ensure scalar log_prob
            while log_prob.dim() > 0:
                log_prob = log_prob.sum()
        else:  # Pong (discrete)
            action_for_env = (
                action.item() if action.dim() == 0 else action.squeeze().item()
            )
            log_prob = dist.log_prob(action)
            # Ensure scalar log_prob
            while log_prob.dim() > 0:
                log_prob = log_prob.sum()

        value = value_fn(obs_tensor).item()

        observation, reward, terminated, truncated, info = env.step(action_for_env)
        done = terminated or truncated

        observations.append(obs)
        # Store the action consistently
        if (
            args.env == 2
        ):  # CartPole - store the original continuous action for training
            actions.append(action.squeeze(0).cpu().numpy())
        elif args.env == 1:  # Pong - discrete
            actions.append(
                action.item() if action.dim() == 0 else action.squeeze().item()
            )
        else:  # Continuous environments
            actions.append(action.squeeze(0).cpu().numpy())
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob.item())

        episode_return += reward
        episode_length += 1
        timesteps_collected += 1

        obs = observation
        if done:
            episode_returns.append(episode_return)
            episode_return = 0
            episode_length = 0
            obs = env.reset()[0]

    # Append value for last step for bootstrapping
    values.append(
        value_fn(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)).item()
    )

    return {
        "observations": torch.tensor(np.array(observations), dtype=torch.float32),
        "actions": torch.tensor(np.array(actions), dtype=torch.float32),
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "log_probs": torch.tensor(log_probs, dtype=torch.float32),
        "episode_returns": episode_returns,
    }


def main():
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        required=True,
        help="Environments to use: "
        "1: PongNoFrameskip-v4, "
        "2: CartPole-v1, "
        "3: Pendulum-v1, "
        "4: Ant-v4, "
        "5: Humanoid-v5, "
        "6: CartPole_own",
    )

    parser.add_argument(
        "--trust_region",
        type=int,
        choices=[1, 2],
        required=True,
        help="Choose algorithm: 1: TRPO, 2: PPO",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--config", type=str, default="hyperparameters.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Generate random seed if none provided
    if args.seed is None:
        args.seed = secrets.randbelow(2**32)

    print(f"Using seed: {args.seed}")

    # Load hyperparameters from YAML
    hyperparams, env_settings, general_settings = load_hyperparameters(
        args.env, args.trust_region
    )

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Environment setup
    env_map = {
        1: ("PongNoFrameskip-v4", "pong"),
        2: ("CartPole-v1", "cartpole"),
        3: ("Pendulum-v1", "pendulum"),
        4: ("Ant-v4", "ant"),
        5: ("Humanoid-v5", "humanoid"),
        6: ("CartPole_own", "cartpole_own"),
    }

    env_gym_name, env_name = env_map[args.env]

    if args.env == 6:
        env = MuJoCoCartPoleEnv()
    else:
        env = gym.make(env_gym_name)

    # Algorithm name
    trust_region_name = "trpo" if args.trust_region == 1 else "ppo"

    # Set environment seeds
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # Initialize wandb with loaded hyperparameters and seed info
    wandb.init(
        project=general_settings["wandb_project"],
        name=f"{trust_region_name}_{env_name}_lambda_{hyperparams['lambda_']}_gamma_{hyperparams['gamma']}_seed_{args.seed}",
        config={
            **hyperparams,
            "seed": args.seed,
            "environment": env_name,
            "algorithm": trust_region_name,
        },
    )

    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup based on environment
    network_config = env_settings["networks"][env_name]

    if network_config["policy_type"] == "cnn_discrete":
        env = FrameStacker(
            env, network_config["frame_stack"], preprocess_fn=preprocess_frame
        )
        obs_shape = env.observation_space.shape  # (channels, height, width)
        input_channels = obs_shape[0]  # Extract number of channels (4)
        num_actions = env.action_space.n
        policy = CNNDiscretePolicy(input_channels, num_actions).to(device)
        value_fn = CNNValueFunction(input_channels).to(device)
    elif network_config["policy_type"] == "linear":
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):  # Discrete
            act_dim = env.action_space.n
        else:  # Continuous
            act_dim = env.action_space.shape[0]
        policy = LinearPolicy(obs_dim, act_dim).to(device)
        value_fn = ValueFunction(obs_dim).to(device)
    elif network_config["policy_type"] == "mlp":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        policy = MLPPolicy(obs_dim, act_dim).to(device)
        value_fn = MLPValueFunction(obs_dim).to(device)

    # Optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=config.value_lr)

    best_avg_return = -float("inf")

    print(f"Starting training: {trust_region_name.upper()} on {env_name}")
    print(f"Hyperparameters: {dict(config)}")

    for iteration in range(config.max_iters):
        data = collect_episodes(
            args,
            env,
            policy,
            value_fn,
            config.max_timesteps_per_batch,
            config.gamma,
            device,
        )

        advantages = compute_gae(
            data["rewards"], data["values"], data["dones"], config.gamma, config.lambda_
        )
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + torch.tensor(
            data["values"][:-1], dtype=torch.float32
        ).to(device)

        # Move tensors to device
        observations = data["observations"].to(device)
        actions = data["actions"].to(device)
        old_log_probs = data["log_probs"].clone().detach().to(device)

        # Policy update
        if args.trust_region == 1:  # TRPO
            trpo_update(
                policy,
                observations,
                actions,
                advantages,
                old_log_probs,
                max_kl=config.max_kl,
            )
        elif args.trust_region == 2:  # PPO
            ppo_update(
                policy,
                value_fn,
                policy_optimizer,
                observations,
                actions,
                advantages,
                returns,
                old_log_probs,
                config,
            )

        # Value function update
        value_epochs = getattr(config, "value_epochs", 5)
        update_value_function(
            value_fn, value_optimizer, observations, returns, epochs=value_epochs
        )

        avg_return = np.mean(data["episode_returns"]) if data["episode_returns"] else 0

        # Enhanced logging with seed tracking
        wandb.log(
            {
                "avg_return": avg_return,
                "iteration": iteration,
                "episodes": len(data["episode_returns"]),
                "timesteps": len(data["observations"]),
                "seed": args.seed,
            }
        )

        print(
            f"Iter {iteration} | Avg Return: {avg_return:.2f} | Episodes: {len(data['episode_returns'])} | Seed: {args.seed}"
        )

        # Save best model with seed info
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "value_fn_state_dict": value_fn.state_dict(),
                    "optimizer_state_dict": value_optimizer.state_dict(),
                    "iteration": iteration,
                    "avg_return": avg_return,
                    "hyperparams": dict(config),
                    "seed": args.seed,
                    "environment": env_name,
                    "algorithm": trust_region_name,
                },
                os.path.join(
                    checkpoint_dir,
                    f"{trust_region_name}_best_{env_name}_seed_{args.seed}.pth",
                ),
            )
            print(
                f"Saved new best model at iteration {iteration} with avg return {avg_return:.2f} (seed: {args.seed})"
            )

        # Periodic checkpoint saving with seed info
        if (
            iteration % general_settings.get("save_frequency", 50) == 0
            and iteration > 0
        ):
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "value_fn_state_dict": value_fn.state_dict(),
                    "optimizer_state_dict": value_optimizer.state_dict(),
                    "iteration": iteration,
                    "avg_return": avg_return,
                    "hyperparams": dict(config),
                    "seed": args.seed,
                    "environment": env_name,
                    "algorithm": trust_region_name,
                },
                os.path.join(
                    checkpoint_dir,
                    f"{trust_region_name}_{env_name}_iter_{iteration}_seed_{args.seed}.pth",
                ),
            )
            print(f"Saved checkpoint at iteration {iteration} (seed: {args.seed})")


if __name__ == "__main__":
    main()
