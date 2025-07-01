import torch
import numpy as np
import gymnasium as gym
import ale_py
import os
import cv2
import argparse
import yaml
from models import (
    LinearPolicy,
    ValueFunction,
    MLPPolicy,
    MLPValueFunction,
    CNNDiscretePolicy,
    CNNValueFunction,
)
from mujoco_env import MuJoCoCartPoleEnv
from utils import preprocess_frame, FrameStacker


def load_network_config(env_name):
    """Load network configuration from YAML"""
    try:
        with open("hyperparameters.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config["environment_settings"]["networks"][env_name]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading config: {e}")
        return None


def load_model(filepath, env_name, env, device):
    """Load model from checkpoint with automatic architecture detection"""
    checkpoint = torch.load(filepath, map_location=device)

    # Get network configuration
    network_config = load_network_config(env_name)
    if network_config is None:
        print(f"Could not load network config for {env_name}, using defaults")

    # Determine model architecture based on environment
    if network_config and network_config["policy_type"] == "cnn_discrete":
        # CNN for image-based environments (Pong)
        obs_shape = env.observation_space.shape  # (channels, height, width)
        input_channels = obs_shape[0]  # Extract number of channels
        act_dim = env.action_space.n
        policy = CNNDiscretePolicy(input_channels, act_dim).to(device)
        value_fn = CNNValueFunction(input_channels).to(device)
    elif network_config and network_config["policy_type"] == "linear_discrete":
        # Linear discrete for simple discrete environments (CartPole)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        policy = LinearPolicy(obs_dim, act_dim).to(device)
        value_fn = ValueFunction(obs_dim).to(device)
    elif network_config and network_config["policy_type"] == "linear":
        # Linear for environments (both discrete and continuous)
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):  # Discrete
            act_dim = env.action_space.n
        else:  # Continuous
            act_dim = env.action_space.shape[0]
        policy = LinearPolicy(obs_dim, act_dim).to(device)
        value_fn = ValueFunction(obs_dim).to(device)
    elif network_config and network_config["policy_type"] == "mlp":
        # MLP for complex continuous environments (Ant, Humanoid)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        policy = MLPPolicy(obs_dim, act_dim).to(device)
        value_fn = MLPValueFunction(obs_dim).to(device)
    else:
        # Fallback: auto-detect based on environment complexity
        if (
            hasattr(env.observation_space, "shape")
            and len(env.observation_space.shape) > 1
        ):
            # Image-based environment
            obs_shape = env.observation_space.shape
            input_channels = obs_shape[0]
            act_dim = env.action_space.n
            policy = CNNDiscretePolicy(input_channels, act_dim).to(device)
            value_fn = CNNValueFunction(input_channels).to(device)
        else:
            obs_dim = env.observation_space.shape[0]
            if hasattr(env.action_space, "n"):  # Discrete
                act_dim = env.action_space.n
                policy = LinearPolicy(obs_dim, act_dim).to(device)
                value_fn = ValueFunction(obs_dim).to(device)
            else:  # Continuous
                act_dim = env.action_space.shape[0]
                if obs_dim > 10:  # Complex environment
                    policy = MLPPolicy(obs_dim, act_dim).to(device)
                    value_fn = MLPValueFunction(obs_dim).to(device)
                else:  # Simple environment
                    policy = LinearPolicy(obs_dim, act_dim).to(device)
                    value_fn = ValueFunction(obs_dim).to(device)

    # Load the saved weights
    policy.load_state_dict(checkpoint["policy_state_dict"])
    value_fn.load_state_dict(checkpoint["value_fn_state_dict"])

    policy.eval()
    value_fn.eval()

    return policy, value_fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "--model", type=str, required=True, help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["none", "human", "video"],
        default="none",
        help="Rendering mode: 'none' (no rendering), 'human' (display window), 'video' (save to file)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for evaluation"
    )

    args = parser.parse_args()

    # Set seeds for reproducible evaluation
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment setup matching main.py
    env_map = {
        1: ("PongNoFrameskip-v4", "pong"),
        2: ("CartPole-v1", "cartpole"),
        3: ("Pendulum-v1", "pendulum"),
        4: ("Ant-v4", "ant"),
        5: ("Humanoid-v5", "humanoid"),
        6: ("CartPole_own", "cartpole_own"),
    }

    env_gym_name, env_name = env_map[args.env]

    # Create environment
    if args.env == 6:
        env = MuJoCoCartPoleEnv()
        # Load network configuration first for CartPole_own
        network_config = load_network_config(env_name)
    else:
        # Set up rendering based on user choice
        mode = ""
        fourcc = None
        width = height = 0

        if args.render == "human":
            try:
                env = gym.make(env_gym_name, render_mode="human")
                print("Using human rendering mode")
                mode = "human"
            except Exception as e:
                print(f"Human rendering failed ({e}), falling back to no rendering")
                env = gym.make(env_gym_name)
                mode = ""
        elif args.render == "video":
            try:
                # For MuJoCo environments, try offscreen rendering
                if args.env in [4, 5]:  # Ant, Humanoid
                    import mujoco

                    env = gym.make(
                        env_gym_name, render_mode="rgb_array", width=640, height=480
                    )
                else:
                    env = gym.make(env_gym_name, render_mode="rgb_array")

                print("Using video recording mode")
                mode = "rgb_array"

                # Get frame dimensions for video recording from the BASE environment
                temp_obs, _ = env.reset()
                frame = env.render()
                if frame is not None:
                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    os.makedirs("videos", exist_ok=True)
                    print(f"Video will be saved with dimensions: {width}x{height}")
                else:
                    raise Exception("Could not get frame from environment")
            except Exception as e:
                print(f"Video rendering failed ({e}), falling back to no rendering")
                env = gym.make(env_gym_name)
                mode = ""
        else:  # args.render == "none"
            env = gym.make(env_gym_name)
            mode = ""

        # Load network configuration AFTER setting up the base environment
        network_config = load_network_config(env_name)

        # Apply frame stacking for CNN-based environments AFTER rendering setup
        # This way the agent gets stacked frames but rendering shows the original environment
        if network_config and network_config["policy_type"] == "cnn_discrete":
            env = FrameStacker(
                env, network_config["frame_stack"], preprocess_fn=preprocess_frame
            )

    # Load the trained model
    print(f"Loading model from: {args.model}")
    policy, value_fn = load_model(args.model, env_name, env, device)

    # Load checkpoint info for display
    checkpoint = torch.load(args.model, map_location=device)
    if "seed" in checkpoint:
        print(f"Model was trained with seed: {checkpoint['seed']}")
    if "avg_return" in checkpoint:
        print(f"Model's training avg return: {checkpoint['avg_return']:.2f}")
    if "iteration" in checkpoint:
        print(f"Model was saved at iteration: {checkpoint['iteration']}")

    # Evaluation loop
    episode_rewards = []
    print(f"Evaluating on {env_name} for {args.episodes} episodes...")

    for episode in range(args.episodes):
        # Initialize video writer for this episode if in video mode
        video_writer = None
        if mode == "rgb_array":
            video_filename = (
                f"videos/{env_name}_episode_{episode + 1}_seed_{args.seed}.mp4"
            )
            video_writer = cv2.VideoWriter(
                video_filename, fourcc, 30.0, (width, height)
            )
            print(f"Recording episode {episode + 1} to {video_filename}")

        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Capture frame for video if in video mode
            # For stacked environments, render from the base environment to show original frames
            if mode == "rgb_array" and video_writer is not None:
                if hasattr(env, "env"):  # FrameStacker wrapper
                    frame = env.env.render()  # Render from base environment
                else:
                    frame = env.render()

                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                dist = policy(obs_tensor)
                # For discrete distributions, use the mode (most likely action)
                # For continuous distributions, use the mean
                if hasattr(dist, "probs"):  # Categorical distribution
                    action = dist.probs.argmax(dim=-1)
                elif hasattr(dist, "mean"):  # Normal distribution
                    action = dist.mean
                else:
                    action = dist.sample()

            # Handle action format based on environment type
            if args.env == 2:  # CartPole (discrete)
                # Convert continuous action to discrete for CartPole
                action_continuous = action.squeeze(0).cpu().numpy()
                if hasattr(action_continuous, "__len__") and len(action_continuous) > 0:
                    action_for_env = 1 if action_continuous[0] > 0 else 0
                else:
                    action_for_env = 1 if action_continuous > 0 else 0
            elif args.env in [3, 6]:  # Pendulum and CartPole_own (continuous)
                action_for_env = action.squeeze(0).cpu().numpy()
            elif args.env in [4, 5]:  # Ant, Humanoid (continuous)
                action_for_env = action.squeeze(0).cpu().numpy()
            else:  # Pong (discrete)
                # For discrete environments like Pong, convert to integer
                action_for_env = (
                    action.item() if action.dim() == 0 else action.squeeze().item()
                )

            obs, reward, terminated, truncated, info = env.step(action_for_env)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Safety check for very long episodes
            if step_count > 10000:
                print(f"Episode {episode + 1} exceeded 10000 steps, terminating...")
                break

        # Release video writer after episode is complete
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved for episode {episode + 1}")

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.2f} (steps: {step_count})")

    # Print evaluation statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    print("\n=== Evaluation Results ===")
    print(f"Environment: {env_name}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
