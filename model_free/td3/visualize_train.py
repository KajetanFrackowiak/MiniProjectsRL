import matplotlib.pyplot as plt
import numpy as np
from agent import AgentDDPG, AgentTD3, AgentSAC
import os
import argparse
from scipy.ndimage import uniform_filter1d
import glob


def smooth_curve(y, window_size=50):
    """Smooth the curve using uniform filter for visual clarity."""
    if len(y) < window_size:
        return y
    return uniform_filter1d(y, size=window_size, mode="nearest")


def load_multiple_runs(checkpoint_pattern, agent_class):
    """
    Load multiple training runs for statistical analysis.

    Args:
        checkpoint_pattern: Pattern to match checkpoint files (e.g., "checkpoint_*_DDPG_*.pth")
        agent_class: Agent class to instantiate for loading

    Returns:
        List of (average_returns, episode_rewards) tuples
    """
    runs_data = []
    checkpoint_files = glob.glob(checkpoint_pattern)

    for checkpoint_path in checkpoint_files:
        try:
            # Create dummy agent to load data
            dummy_agent = agent_class(state_dim=1, action_dim=1)
            average_returns, episode_rewards, _, _ = dummy_agent.load(checkpoint_path)
            if average_returns and episode_rewards:
                runs_data.append((average_returns, episode_rewards))
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")

    return runs_data


def plot_learning_curves(
    ddpg_data=None,
    td3_data=None,
    sac_data=None,
    env_name="Environment",
    save_path=None,
    figsize=(10, 6),
    smoothing_window=50,
):
    """
    Create TD3 paper style learning curves with shaded regions representing standard deviation.

    Args:
        ddpg_data: List of (average_returns, episode_rewards) from multiple DDPG runs
        td3_data: List of (average_returns, episode_rewards) from multiple TD3 runs
        sac_data: List of (average_returns, episode_rewards) from multiple SAC runs
        env_name: Name of the environment for plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size tuple
        smoothing_window: Window size for smoothing curves
    """

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-v0_8")  # Use seaborn style for better aesthetics

    colors = {
        "DDPG": "#1f77b4",  # Blue
        "TD3": "#ff7f0e",  # Orange
        "SAC": "#2ca02c",  # Green
    }

    methods_data = {"DDPG": ddpg_data, "TD3": td3_data, "SAC": sac_data}

    for method_name, runs_data in methods_data.items():
        if runs_data is None or len(runs_data) == 0:
            continue

        # Extract all average returns from multiple runs
        all_returns = []
        max_length = 0

        for avg_returns, _ in runs_data:
            if len(avg_returns) > 0:
                all_returns.append(avg_returns)
                max_length = max(max_length, len(avg_returns))

        if len(all_returns) == 0:
            continue

        # Pad shorter runs with their last value
        padded_returns = []
        for returns in all_returns:
            if len(returns) < max_length:
                # Pad with last value
                padded = list(returns) + [returns[-1]] * (max_length - len(returns))
                padded_returns.append(padded)
            else:
                padded_returns.append(returns[:max_length])

        # Convert to numpy array for easier computation
        returns_array = np.array(padded_returns)

        # Calculate mean and standard deviation across runs
        mean_returns = np.mean(returns_array, axis=0)
        std_returns = np.std(returns_array, axis=0)

        # Smooth the curves
        episodes = np.arange(len(mean_returns))
        smoothed_mean = smooth_curve(mean_returns, smoothing_window)
        smoothed_std = smooth_curve(std_returns, smoothing_window)

        # Plot mean curve
        plt.plot(
            episodes,
            smoothed_mean,
            label=method_name,
            color=colors[method_name],
            linewidth=2,
        )

        # Plot shaded region for half standard deviation (as mentioned in TD3 paper)
        plt.fill_between(
            episodes,
            smoothed_mean - smoothed_std / 2,
            smoothed_mean + smoothed_std / 2,
            color=colors[method_name],
            alpha=0.2,
        )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title(f"Learning Curves - {env_name}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def visualize_single_run(
    average_returns, episode_rewards, method_name, env_name, save_path=None
):
    """
    Visualize a single training run.

    Args:
        average_returns: List of average returns over training
        episode_rewards: List of episode rewards
        method_name: Name of the method (DDPG, TD3, SAC)
        env_name: Environment name
        save_path: Path to save the plot
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot average returns
    episodes = np.arange(len(average_returns))
    smoothed_avg = smooth_curve(average_returns, window_size=50)

    ax1.plot(episodes, average_returns, alpha=0.3, color="lightblue", label="Raw")
    ax1.plot(episodes, smoothed_avg, color="blue", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Return (last 100 episodes)")
    ax1.set_title(f"{method_name} - Average Returns - {env_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode rewards
    episodes_rewards = np.arange(len(episode_rewards))
    smoothed_rewards = smooth_curve(episode_rewards, window_size=50)

    ax2.plot(
        episodes_rewards, episode_rewards, alpha=0.3, color="lightcoral", label="Raw"
    )
    ax2.plot(
        episodes_rewards, smoothed_rewards, color="red", linewidth=2, label="Smoothed"
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Reward")
    ax2.set_title(f"{method_name} - Episode Rewards - {env_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize RL training results")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=".",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="Environment",
        help="Environment name for plot title",
    )
    parser.add_argument(
        "--single_run",
        type=str,
        default=None,
        help="Path to single checkpoint file for single run visualization",
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the plot"
    )
    parser.add_argument(
        "--smoothing", type=int, default=50, help="Smoothing window size"
    )

    args = parser.parse_args()

    if args.single_run:
        # Load and visualize single run
        if "DDPG" in args.single_run:
            agent = AgentDDPG(state_dim=1, action_dim=1)
            method_name = "DDPG"
        elif "TD3" in args.single_run:
            agent = AgentTD3(state_dim=1, action_dim=1)
            method_name = "TD3"
        elif "SAC" in args.single_run:
            agent = AgentSAC(state_dim=1, action_dim=1)
            method_name = "SAC"
        else:
            print("Cannot determine method from filename")
            return

        try:
            average_returns, episode_rewards, _, _ = agent.load(args.single_run)
            visualize_single_run(
                average_returns,
                episode_rewards,
                method_name,
                args.env_name,
                args.save_path,
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    else:
        # Load multiple runs for comparison
        ddpg_pattern = os.path.join(args.checkpoint_dir, "*DDPG*.pth")
        td3_pattern = os.path.join(args.checkpoint_dir, "*TD3*.pth")
        sac_pattern = os.path.join(args.checkpoint_dir, "*SAC*.pth")

        ddpg_data = load_multiple_runs(ddpg_pattern, AgentDDPG)
        td3_data = load_multiple_runs(td3_pattern, AgentTD3)
        sac_data = load_multiple_runs(sac_pattern, AgentSAC)

        print(f"Loaded {len(ddpg_data)} DDPG runs")
        print(f"Loaded {len(td3_data)} TD3 runs")
        print(f"Loaded {len(sac_data)} SAC runs")

        plot_learning_curves(
            ddpg_data=ddpg_data,
            td3_data=td3_data,
            sac_data=sac_data,
            env_name=args.env_name,
            save_path=args.save_path,
            smoothing_window=args.smoothing,
        )


if __name__ == "__main__":
    main()
