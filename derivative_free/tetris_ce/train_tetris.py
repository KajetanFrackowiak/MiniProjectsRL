#!/usr/bin/env python3
"""
Main training script for Tetris using the Noisy Cross-Entropy Method.

This script implements the approach from:
Szita, I., & Lőrincz, A. (2006). Learning Tetris using the noisy cross-entropy method.
Neural computation, 18(12), 2936-2941.

Usage:
    python train_tetris.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tetris_environment import TetrisCEMTrainer
from models import LinearTetrisPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Train Tetris using Noisy Cross-Entropy Method"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size for CEM (default: 20)",
    )
    parser.add_argument(
        "--elite-ratio",
        type=float,
        default=0.3,
        help="Elite ratio for CEM (default: 0.3)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="Noise standard deviation (default: 0.5)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations (default: 30)",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=3,
        help="Games per individual evaluation (default: 3)",
    )
    parser.add_argument(
        "--test-games", type=int, default=5, help="Number of test games (default: 5)"
    )
    parser.add_argument(
        "--save-weights",
        type=str,
        default="tetris_weights.npy",
        help="File to save best weights (default: tetris_weights.npy)",
    )

    args = parser.parse_args()

    print("Tetris Training with Noisy Cross-Entropy Method")
    print("=" * 60)
    print(f"Population size: {args.population_size}")
    print(f"Elite ratio: {args.elite_ratio}")
    print(f"Noise std: {args.noise_std}")
    print(f"Generations: {args.generations}")
    print(f"Games per evaluation: {args.games_per_eval}")
    print("=" * 60)

    # Create trainer
    trainer = TetrisCEMTrainer(
        population_size=args.population_size,
        elite_ratio=args.elite_ratio,
        noise_std=args.noise_std,
        max_generations=args.generations,
        games_per_individual=args.games_per_eval,
    )

    try:
        # Train the policy
        print("\nStarting training...")
        best_policy = trainer.train()

        # Test the best policy
        print("\n" + "=" * 60)
        print("Testing best policy...")
        avg_score = trainer.test_policy(best_policy, num_games=args.test_games)

        # Save the weights
        print(f"\nSaving weights to {args.save_weights}...")
        np.save(args.save_weights, best_policy.weights)

        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final average test score: {avg_score:.2f}")
        print(f"Best weights saved to: {args.save_weights}")
        print("\nFinal feature weights:")
        for i, (name, weight) in enumerate(
            zip(best_policy.feature_extractor.feature_names, best_policy.weights)
        ):
            print(f"  {name:20s}: {weight:8.4f}")

        # Plot training history
        stats = trainer.cem.get_stats()
        if stats["score_history"]:
            plt.figure(figsize=(10, 6))
            plt.plot(stats["score_history"])
            plt.title("Tetris Training Progress (Noisy Cross-Entropy Method)")
            plt.xlabel("Generation")
            plt.ylabel("Best Score")
            plt.grid(True)
            plt.savefig("tetris_training_progress.png", dpi=150, bbox_inches="tight")
            print("Training plot saved as: tetris_training_progress.png")
            plt.show()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if trainer.cem.best_weights is not None:
            print("Saving best weights found so far...")
            best_policy = trainer.cem.get_best_policy()
            np.save(args.save_weights, best_policy.weights)
            print(f"Weights saved to: {args.save_weights}")

    except Exception as e:
        print(f"\nError during training: {e}")
        print("Check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1

    return 0


def load_and_test_policy(weights_file: str, num_games: int = 10):
    """
    Load a saved policy and test it.

    Args:
        weights_file: Path to saved weights file
        num_games: Number of games to test
    """
    try:
        weights = np.load(weights_file)
        print(f"Loaded weights from {weights_file}")

        # Create policy and set weights
        policy = LinearTetrisPolicy()
        policy.set_weights(weights)

        # Create trainer for testing
        trainer = TetrisCEMTrainer()

        print(f"Testing policy over {num_games} games...")
        avg_score = trainer.test_policy(policy, num_games=num_games)

        print(f"Average score: {avg_score:.2f}")

        return avg_score

    except FileNotFoundError:
        print(f"Weights file {weights_file} not found.")
        return None
    except Exception as e:
        print(f"Error loading/testing policy: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - load and test existing weights
        weights_file = sys.argv[2] if len(sys.argv) > 2 else "tetris_weights.npy"
        num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        load_and_test_policy(weights_file, num_games)
    else:
        # Training mode
        sys.exit(main())
