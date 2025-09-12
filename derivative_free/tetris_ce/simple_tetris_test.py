#!/usr/bin/env python3
"""
Simplified Tetris environment using board state features directly.
This avoids the complexity of visual processing and focuses on testing CEM.
"""

import numpy as np
from models import TetrisFeatureExtractor, LinearTetrisPolicy, NoisyCrossEntropyMethod


class SimpleTetrisEnvironment:
    """
    Simplified Tetris environment that works with board states directly.
    This is much easier to debug and verify than visual processing.
    """

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.feature_extractor = TetrisFeatureExtractor()
        self.reset()

        # Simple tetris pieces (just use I-piece for simplicity)
        self.pieces = [
            np.array([[1, 1, 1, 1]]),  # I-piece horizontal
            np.array([[1], [1], [1], [1]]),  # I-piece vertical
        ]

    def reset(self):
        """Reset the game."""
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.steps = 0
        return self.get_features()

    def get_features(self):
        """Get current board features."""
        if self.game_over:
            return np.zeros(6)

        piece_info = {
            "landing_height": 0,
            "lines_cleared": self.lines_cleared,
            "piece_cells_in_cleared_lines": 0,
        }

        return self.feature_extractor.extract_features(self.board, piece_info)

    def step(self, action):
        """
        Take a step. Actions:
        0: Place piece at column 0
        1: Place piece at column 1
        ...
        9: Place piece at column 9
        """
        if self.game_over:
            return self.get_features(), 0, True, False, {}

        self.steps += 1

        # Simple action: try to place an I-piece (4 units) at the specified column
        piece = self.pieces[0]  # Use horizontal I-piece
        col = action % self.width

        # Make sure piece fits
        if col + piece.shape[1] > self.width:
            col = self.width - piece.shape[1]

        # Find where piece would land
        landing_row = self._find_landing_row(piece, col)

        if landing_row < 0:
            # Can't place piece - game over
            self.game_over = True
            return self.get_features(), -10, True, False, {}

        # Place the piece
        self._place_piece(piece, landing_row, col)

        # Clear lines and calculate reward
        lines_cleared = self._clear_lines()
        reward = self._calculate_reward(lines_cleared)

        # Check if game over (board full)
        if np.any(self.board[0, :]):
            self.game_over = True
            reward -= 10

        # End episode after too many steps
        truncated = self.steps >= 100

        return self.get_features(), reward, self.game_over, truncated, {}

    def _find_landing_row(self, piece, col):
        """Find where piece would land."""
        piece_height, piece_width = piece.shape

        for row in range(self.height - piece_height, -1, -1):
            if self._can_place_piece(piece, row, col):
                return row
        return -1  # Can't place

    def _can_place_piece(self, piece, row, col):
        """Check if piece can be placed at position."""
        piece_height, piece_width = piece.shape

        if row < 0 or row + piece_height > self.height:
            return False
        if col < 0 or col + piece_width > self.width:
            return False

        # Check for collisions
        for pr in range(piece_height):
            for pc in range(piece_width):
                if piece[pr, pc] and self.board[row + pr, col + pc]:
                    return False

        return True

    def _place_piece(self, piece, row, col):
        """Place piece on board."""
        piece_height, piece_width = piece.shape

        for pr in range(piece_height):
            for pc in range(piece_width):
                if piece[pr, pc]:
                    self.board[row + pr, col + pc] = 1

    def _clear_lines(self):
        """Clear completed lines and return number cleared."""
        lines_cleared = 0
        row = self.height - 1

        while row >= 0:
            if np.all(self.board[row, :]):
                # Clear this line
                self.board[1 : row + 1, :] = self.board[:row, :]
                self.board[0, :] = 0
                lines_cleared += 1
                # Don't decrement row since we moved lines down
            else:
                row -= 1

        self.lines_cleared = lines_cleared
        return lines_cleared

    def _calculate_reward(self, lines_cleared):
        """Calculate reward based on lines cleared."""
        if lines_cleared == 0:
            return 1  # Small reward for surviving
        elif lines_cleared == 1:
            return 10
        elif lines_cleared == 2:
            return 30
        elif lines_cleared == 3:
            return 60
        else:
            return 100  # Tetris!


class SimpleTetrisCEMTrainer:
    """CEM trainer for simplified Tetris."""

    def __init__(self, population_size=30, elite_ratio=0.3, generations=20):
        self.cem = NoisyCrossEntropyMethod(
            num_features=6,
            population_size=population_size,
            elite_ratio=elite_ratio,
            noise_std=0.1,
            initial_std=1.0,
        )
        self.generations = generations

    def evaluate_policy(self, weights, num_games=5):
        """Evaluate a policy over multiple games."""
        policy = LinearTetrisPolicy()
        policy.set_weights(weights)

        total_score = 0

        for game in range(num_games):
            env = SimpleTetrisEnvironment()
            features = env.reset()
            game_score = 0

            while not env.game_over and env.steps < 100:
                # Use policy to evaluate all possible actions
                action_scores = []

                for action in range(10):  # 10 possible columns
                    action_scores.append(policy.evaluate_position(features))

                # Choose action with best score (plus some randomness)
                best_action = np.argmax(action_scores)

                features, reward, done, truncated, info = env.step(best_action)
                game_score += reward

                if done or truncated:
                    break

            total_score += game_score

        return total_score / num_games

    def train(self):
        """Train using CEM."""
        print("Starting Simplified Tetris CEM Training")
        print("=" * 40)

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            # Sample population
            population = self.cem.sample_population()

            # Evaluate
            scores = []
            for i, weights in enumerate(population):
                score = self.evaluate_policy(weights, num_games=3)
                scores.append(score)
                print(f"  Individual {i + 1:2d}: Score = {score:6.1f}")

            scores = np.array(scores)

            # Update distribution
            self.cem.update_distribution(population, scores)

            # Stats
            stats = self.cem.get_stats()
            print(f"  Best this gen: {np.max(scores):.1f}")
            print(f"  Best overall: {stats['best_score']:.1f}")

        # Get best policy
        best_policy = LinearTetrisPolicy()
        best_policy.set_weights(self.cem.best_weights)
        return best_policy


def test_simple_tetris():
    """Test the simplified Tetris environment."""
    print("Testing Simplified Tetris Environment")
    print("=" * 40)

    env = SimpleTetrisEnvironment()
    features = env.reset()

    print(f"Initial features: {features}")
    print(f"Initial board:\n{env.board}")

    # Test a few random actions
    for step in range(10):
        action = np.random.randint(0, 10)
        features, reward, done, truncated, info = env.step(action)

        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Features: {features}")
        print(f"  Game over: {done}")

        if done or truncated:
            print("Game ended!")
            break

    print(f"\nFinal board:\n{env.board}")
    print(f"Final score: {env.score}")


def main():
    """Main function."""
    # Test environment
    test_simple_tetris()

    print("\n" + "=" * 50)

    # Train with CEM
    trainer = SimpleTetrisCEMTrainer(population_size=20, generations=10)
    best_policy = trainer.train()

    print(f"\nBest weights: {best_policy.weights}")

    # Test best policy
    print("\nTesting best policy:")
    final_score = trainer.evaluate_policy(best_policy.weights, num_games=10)
    print(f"Average score with best policy: {final_score:.1f}")


if __name__ == "__main__":
    main()
