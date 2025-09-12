import gymnasium as gym
import ale_py
import numpy as np
import cv2
from models import TetrisFeatureExtractor, LinearTetrisPolicy, NoisyCrossEntropyMethod


class AtariTetrisWrapper:
    """
    Wrapper for Atari Tetris that adapts it for use with the Noisy Cross-Entropy Method.
    Converts visual observations to board states and extracts features.
    """

    def __init__(self, env_id: str = "ALE/Tetris-v5"):
        self.env = gym.make(env_id)
        self.feature_extractor = TetrisFeatureExtractor()

        # Board dimensions for standard Tetris
        self.board_height = 20
        self.board_width = 10

        # For tracking game state
        self.last_board = None
        self.last_score = 0
        self.lines_cleared_this_step = 0

    def reset(self):
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset()
        self.last_board = self._extract_board_from_observation(obs)
        self.last_score = 0
        self.lines_cleared_this_step = 0
        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract current board state
        current_board = self._extract_board_from_observation(obs)

        # Calculate lines cleared (approximate from reward change)
        self.lines_cleared_this_step = self._estimate_lines_cleared(reward)

        self.last_board = current_board
        self.last_score += reward

        return obs, reward, done, truncated, info

    def _extract_board_from_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract board state from Atari visual observation.
        This is a simplified version - in practice, you'd need more sophisticated
        image processing to accurately extract the Tetris board state.
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Resize to standard Tetris board dimensions
        # This is a placeholder - real implementation would need careful calibration
        board_region = gray[50:190, 80:120]  # Approximate board region
        resized = cv2.resize(board_region, (self.board_width, self.board_height))

        # Convert to binary (0=empty, 1=filled)
        # Threshold may need adjustment based on Atari Tetris graphics
        _, binary = cv2.threshold(resized, 50, 1, cv2.THRESH_BINARY)

        return binary.astype(np.int32)

    def _estimate_lines_cleared(self, reward: float) -> int:
        """
        Estimate number of lines cleared based on reward.
        Atari Tetris typically gives different scores for different line clears.
        """
        if reward == 0:
            return 0
        elif reward <= 40:  # Single line
            return 1
        elif reward <= 100:  # Double line
            return 2
        elif reward <= 300:  # Triple line
            return 3
        else:  # Tetris (4 lines)
            return 4

    def get_current_features(self, landing_height: int = 0) -> np.ndarray:
        """
        Extract features from current board state.

        Args:
            landing_height: Height where the last piece landed
        """
        if self.last_board is None:
            return np.zeros(6)

        piece_info = {
            "landing_height": landing_height,
            "lines_cleared": self.lines_cleared_this_step,
            "piece_cells_in_cleared_lines": 0,  # Simplified
        }

        return self.feature_extractor.extract_features(self.last_board, piece_info)

    def close(self):
        """Close the environment."""
        self.env.close()


class TetrisCEMTrainer:
    """
    Trainer for Tetris using the Noisy Cross-Entropy Method with Atari environment.
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_ratio: float = 0.2,
        noise_std: float = 0.5,
        max_generations: int = 100,
        games_per_individual: int = 5,
    ):
        """
        Initialize the CEM trainer.

        Args:
            population_size: Number of individuals in each generation
            elite_ratio: Fraction of population to use as elite
            noise_std: Standard deviation of noise added to elite
            max_generations: Maximum number of generations to run
            games_per_individual: Number of games to play per individual for evaluation
        """
        self.cem = NoisyCrossEntropyMethod(
            num_features=6,
            population_size=population_size,
            elite_ratio=elite_ratio,
            noise_std=noise_std,
        )
        self.max_generations = max_generations
        self.games_per_individual = games_per_individual

    def evaluate_individual(self, weights: np.ndarray) -> float:
        """
        Evaluate a single individual (weight vector) by playing Tetris games.

        Args:
            weights: Weight vector for the linear policy

        Returns:
            Average score over multiple games
        """
        policy = LinearTetrisPolicy()
        policy.set_weights(weights)

        total_score = 0

        for game in range(self.games_per_individual):
            env_wrapper = AtariTetrisWrapper()
            obs, _ = env_wrapper.reset()

            episode_score = 0
            done = False
            steps = 0
            max_steps = 1000  # Prevent infinite games

            while not done and steps < max_steps:
                # Get current features
                features = env_wrapper.get_current_features()

                # Simple action selection based on features
                # In practice, you'd need to generate possible actions and evaluate them
                action = self._select_action_from_features(features, policy)

                obs, reward, done, truncated, info = env_wrapper.step(action)
                episode_score += reward
                steps += 1

                if truncated:
                    done = True

            total_score += episode_score
            env_wrapper.close()

        return total_score / self.games_per_individual

    def _select_action_from_features(
        self, features: np.ndarray, policy: LinearTetrisPolicy
    ) -> int:
        """
        Convert features to action selection.
        This is simplified - in practice, you'd evaluate all possible piece placements.
        """
        # Evaluate the current position
        score = policy.evaluate_position(features)

        # Simple heuristic action selection based on score
        # This is a placeholder - real implementation would be more sophisticated
        if score > 0:
            return 2  # Move right
        elif score < -10:
            return 1  # Move left
        elif features[4] > 5:  # Too many holes
            return 4  # Rotate
        else:
            return 3  # Move down

    def train(self) -> LinearTetrisPolicy:
        """
        Train the Tetris policy using CEM.

        Returns:
            Best policy found
        """
        print("Starting Noisy Cross-Entropy Method training for Tetris...")

        for generation in range(self.max_generations):
            print(f"\nGeneration {generation + 1}/{self.max_generations}")

            # Sample population
            population = self.cem.sample_population()

            # Evaluate all individuals
            scores = []
            for i, weights in enumerate(population):
                score = self.evaluate_individual(weights)
                scores.append(score)
                print(f"  Individual {i + 1:2d}: Score = {score:6.1f}")

            scores = np.array(scores)

            # Update distribution
            self.cem.update_distribution(population, scores)

            # Print statistics
            stats = self.cem.get_stats()
            print(f"  Best score this generation: {np.max(scores):.1f}")
            print(f"  Best score overall: {stats['best_score']:.1f}")
            print(f"  Mean weights: {stats['current_mean']}")
            print(f"  Std weights: {stats['current_std']}")

        return self.cem.get_best_policy()

    def test_policy(self, policy: LinearTetrisPolicy, num_games: int = 10) -> float:
        """
        Test a trained policy.

        Args:
            policy: The policy to test
            num_games: Number of games to play for testing

        Returns:
            Average score
        """
        total_score = 0

        for game in range(num_games):
            env_wrapper = AtariTetrisWrapper()
            obs, _ = env_wrapper.reset()

            episode_score = 0
            done = False
            steps = 0
            max_steps = 2000

            while not done and steps < max_steps:
                features = env_wrapper.get_current_features()
                action = self._select_action_from_features(features, policy)

                obs, reward, done, truncated, info = env_wrapper.step(action)
                episode_score += reward
                steps += 1

                if truncated:
                    done = True

            total_score += episode_score
            env_wrapper.close()
            print(f"Test game {game + 1}: Score = {episode_score}")

        avg_score = total_score / num_games
        print(f"Average test score: {avg_score:.1f}")
        return avg_score


def main():
    """Main training loop."""
    # Create trainer
    trainer = TetrisCEMTrainer(
        population_size=20,  # Smaller for faster testing
        elite_ratio=0.3,
        noise_std=0.5,
        max_generations=50,
        games_per_individual=3,
    )

    # Train the policy
    best_policy = trainer.train()

    # Test the best policy
    print("\n" + "=" * 50)
    print("Testing best policy...")
    trainer.test_policy(best_policy, num_games=5)

    # Print final weights
    print(f"\nFinal weights: {best_policy.weights}")
    print(f"Feature names: {best_policy.feature_extractor.feature_names}")


if __name__ == "__main__":
    main()
