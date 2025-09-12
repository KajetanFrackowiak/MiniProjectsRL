import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import ale_py
from models import NoisyCrossEntropyMethod


class SimpleCEMPolicy:
    """
    Simple linear policy for continuous control tasks like CartPole and Pendulum.
    """

    def __init__(self, obs_dim: int, act_dim: int, discrete: bool = True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete
        self.weights = np.random.randn(obs_dim, act_dim) * 0.1

    def set_weights(self, weights: np.ndarray):
        """Set the policy weights."""
        self.weights = weights.reshape(self.obs_dim, self.act_dim)

    def get_weights_flat(self) -> np.ndarray:
        """Get flattened weights."""
        return self.weights.flatten()

    def act(self, obs: np.ndarray) -> int:
        """Select action based on observation."""
        if self.discrete:
            # For discrete actions (CartPole)
            logits = np.dot(obs, self.weights)
            if self.act_dim == 1:
                return int(logits[0] > 0)  # Binary action
            else:
                return np.argmax(logits)
        else:
            # For continuous actions (Pendulum)
            action = np.dot(obs, self.weights)
            return np.clip(action, -2.0, 2.0)  # Clip to valid range


class SimpleCEMTrainer:
    """
    Simplified CEM trainer for CartPole and Pendulum environments.
    """

    def __init__(
        self, env_name: str, population_size: int = 50, elite_ratio: float = 0.2
    ):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Determine if environment is discrete or continuous
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.discrete = True
            self.act_dim = (
                1 if self.env.action_space.n == 2 else self.env.action_space.n
            )
        else:
            self.discrete = False
            self.act_dim = self.env.action_space.shape[0]

        self.obs_dim = self.env.observation_space.shape[0]

        # Calculate total number of parameters
        self.num_params = self.obs_dim * self.act_dim

        # Initialize CEM
        self.cem = NoisyCrossEntropyMethod(
            num_features=self.num_params,
            population_size=population_size,
            elite_ratio=elite_ratio,
            noise_std=0.1,
            initial_std=1.0,
        )

        print(f"Environment: {env_name}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.act_dim}")
        print(f"Discrete actions: {self.discrete}")
        print(f"Total parameters: {self.num_params}")

    def evaluate_policy(self, weights: np.ndarray, num_episodes: int = 3) -> float:
        """Evaluate a policy over multiple episodes."""
        policy = SimpleCEMPolicy(self.obs_dim, self.act_dim, self.discrete)
        policy.set_weights(weights)

        total_reward = 0

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 1000

            while not done and steps < max_steps:
                action = policy.act(obs)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                steps += 1

                if truncated:
                    done = True

            total_reward += episode_reward

        return total_reward / num_episodes

    def train(self, num_generations: int = 50) -> SimpleCEMPolicy:
        """Train using CEM."""
        scores_history = []

        print(f"Training on {self.env_name} for {num_generations} generations...")

        for generation in range(num_generations):
            # Sample population
            population = self.cem.sample_population()

            # Evaluate all individuals
            scores = []
            for i, weights in enumerate(population):
                score = self.evaluate_policy(weights)
                scores.append(score)

            scores = np.array(scores)
            scores_history.append(np.max(scores))

            # Update CEM distribution
            self.cem.update_distribution(population, scores)

            # Print progress
            if generation % 10 == 0 or generation == num_generations - 1:
                stats = self.cem.get_stats()
                print(
                    f"Generation {generation:3d}: Best={np.max(scores):8.2f}, "
                    f"Mean={np.mean(scores):8.2f}, Overall Best={stats['best_score']:8.2f}"
                )

        # Create best policy
        best_policy = SimpleCEMPolicy(self.obs_dim, self.act_dim, self.discrete)
        best_policy.set_weights(self.cem.best_weights)

        # Skip plotting for now to avoid hanging
        print("Training completed! (Plotting disabled to prevent hanging)")
        # TODO: Fix plotting issue later

        return best_policy

    def test_policy(
        self, policy: SimpleCEMPolicy, num_episodes: int = 10, render: bool = False
    ):
        """Test a trained policy."""
        if render:
            test_env = gym.make(self.env_name, render_mode="human")
        else:
            test_env = self.env

        scores = []

        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 1000:
                action = policy.act(obs)
                obs, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                steps += 1

                # Explicitly render if we're in rendering mode
                if render:
                    test_env.render()
                    time.sleep(0.02)  # Small delay to make rendering visible

                if truncated:
                    done = True

            scores.append(total_reward)
            print(f"Episode {episode + 1}: Score = {total_reward:.2f}")

        avg_score = np.mean(scores)
        print(f"Average score over {num_episodes} episodes: {avg_score:.2f}")

        if render:
            test_env.close()

        return avg_score


def train_cartpole():
    """Train on CartPole environment."""
    trainer = SimpleCEMTrainer("CartPole-v1", population_size=30, elite_ratio=0.3)
    best_policy = trainer.train(num_generations=30)

    print("\nTesting best policy on CartPole:")
    trainer.test_policy(best_policy, num_episodes=5)

    return best_policy


def train_pendulum():
    """Train on Pendulum environment."""
    trainer = SimpleCEMTrainer("Pendulum-v1", population_size=50, elite_ratio=0.2)
    best_policy = trainer.train(num_generations=200)

    print("\nTesting best policy on Pendulum:")
    trainer.test_policy(best_policy, num_episodes=5, render=True)

    return best_policy


def main():
    """Main function to demonstrate CEM on different environments."""
    print("Cross-Entropy Method Demo")
    print("=" * 50)

    # Train on CartPole
    print("\n1. Training on CartPole-v1...")
    cartpole_policy = train_cartpole()

    print("\n" + "=" * 50)

    # Train on Pendulum
    print("\n2. Training on Pendulum-v1...")
    # pendulum_policy = train_pendulum()

    print("\n" + "=" * 50)
    print("Training completed!")

    # Optional: Save trained policies
    print("\nSaving trained policies...")
    np.save("cartpole_weights.npy", cartpole_policy.get_weights_flat())
    # np.save("pendulum_weights.npy", pendulum_policy.get_weights_flat())
    print("Policies saved as cartpole_weights.npy and pendulum_weights.npy")


def load_and_test_weights(
    weights_file: str, env_name: str, num_episodes: int = 5, render: bool = True
):
    """
    Load saved weights and test them with visual rendering.

    Args:
        weights_file: Path to the saved weights (.npy file)
        env_name: Environment name (e.g., 'CartPole-v1', 'Pendulum-v1')
        num_episodes: Number of episodes to test
        render: Whether to show visual rendering
    """
    try:
        # Load weights
        weights = np.load(weights_file)
        print(f"Loaded weights from {weights_file}")
        print(f"Weights shape: {weights.shape}")
        print(f"Weights: {weights}")

        # Create policy first to determine dimensions
        temp_env = gym.make(env_name)
        if isinstance(temp_env.action_space, gym.spaces.Discrete):
            discrete = True
            act_dim = 1 if temp_env.action_space.n == 2 else temp_env.action_space.n
        else:
            discrete = False
            act_dim = temp_env.action_space.shape[0]

        obs_dim = temp_env.observation_space.shape[0]
        temp_env.close()

        # Create policy and set weights
        policy = SimpleCEMPolicy(obs_dim, act_dim, discrete)
        policy.set_weights(weights)

        # Force render to False for now to avoid freezing
        if render:
            print(
                "Warning: Rendering disabled to prevent freezing. Set render=False to suppress this message."
            )
            render = False

        # Create environment
        if render:
            test_env = gym.make(env_name, render_mode="human")
            print(
                f"\nTesting policy on {env_name} for {num_episodes} episodes with visual rendering..."
            )
        else:
            test_env = gym.make(env_name)
            print(
                f"\nTesting policy on {env_name} for {num_episodes} episodes (no rendering)..."
            )

        print("Press Ctrl+C to stop early if needed")

        scores = []

        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            total_reward = 0
            done = False
            steps = 0

            print(f"Episode {episode + 1} starting...")

            while not done and steps < 1000:
                action = policy.act(obs)
                obs, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                steps += 1

                # Explicitly render if we're in rendering mode
                if render:
                    test_env.render()
                    time.sleep(0.02)  # Small delay to make rendering visible

                if truncated:
                    done = True

            scores.append(total_reward)
            print(f"Episode {episode + 1}: Score = {total_reward:.2f}, Steps = {steps}")

        avg_score = np.mean(scores)
        print(f"\nAverage score over {num_episodes} episodes: {avg_score:.2f}")

        test_env.close()

        return avg_score

    except FileNotFoundError:
        print(f"Error: Weights file '{weights_file}' not found!")
        return None
    except Exception as e:
        print(f"Error testing policy: {e}")
        return None


def test_saved_policies():
    """Test all saved policies with visual rendering."""
    print("Testing Saved Policies")
    print("=" * 50)

    # Test CartPole if weights exist (enable rendering for visual feedback)
    print("\n1. Testing CartPole policy...")
    load_and_test_weights(
        "cartpole_weights.npy", "CartPole-v1", num_episodes=3, render=True
    )

    # Test Pendulum if weights exist
    # print("\n2. Testing Pendulum policy...")
    # load_and_test_weights("pendulum_weights.npy", "Pendulum-v1", num_episodes=3, render=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - load and test existing weights
        test_saved_policies()
    else:
        # Training mode
        main()
