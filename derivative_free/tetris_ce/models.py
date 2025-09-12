import numpy as np
from typing import List, Tuple


class TetrisFeatureExtractor:
    """
    Extracts handcrafted features from Tetris board state as used in Szita & Lőrincz (2006).
    These are the key features mentioned in the original paper.
    """

    def __init__(self):
        self.feature_names = [
            "landing_height",
            "eroded_piece_cells",
            "row_transitions",
            "col_transitions",
            "holes",
            "cumulative_wells",
        ]

    def extract_features(
        self, board: np.ndarray, piece_landing_info: dict
    ) -> np.ndarray:
        """
        Extract the 6 key features used in the original paper.

        Args:
            board: 2D numpy array representing the Tetris board (0=empty, 1=filled)
            piece_landing_info: Dict with 'landing_height', 'lines_cleared', 'piece_cells'

        Returns:
            numpy array of features
        """
        features = np.zeros(6)

        # Feature 1: Landing Height
        features[0] = piece_landing_info.get("landing_height", 0)

        # Feature 2: Eroded Piece Cells
        lines_cleared = piece_landing_info.get("lines_cleared", 0)
        piece_cells_in_cleared_lines = piece_landing_info.get(
            "piece_cells_in_cleared_lines", 0
        )
        features[1] = lines_cleared * piece_cells_in_cleared_lines

        # Feature 3: Row Transitions
        features[2] = self._count_row_transitions(board)

        # Feature 4: Column Transitions
        features[3] = self._count_col_transitions(board)

        # Feature 5: Holes
        features[4] = self._count_holes(board)

        # Feature 6: Cumulative Wells
        features[5] = self._count_cumulative_wells(board)

        return features

    def _count_row_transitions(self, board: np.ndarray) -> int:
        """Count horizontal transitions between filled and empty cells."""
        transitions = 0
        rows, cols = board.shape

        for r in range(rows):
            # Add boundary transitions
            if board[r, 0] == 0:
                transitions += 1
            if board[r, -1] == 0:
                transitions += 1

            # Count internal transitions
            for c in range(cols - 1):
                if board[r, c] != board[r, c + 1]:
                    transitions += 1

        return transitions

    def _count_col_transitions(self, board: np.ndarray) -> int:
        """Count vertical transitions between filled and empty cells."""
        transitions = 0
        rows, cols = board.shape

        for c in range(cols):
            # Add boundary transition (top is considered filled)
            if board[0, c] == 0:
                transitions += 1

            # Count internal transitions
            for r in range(rows - 1):
                if board[r, c] != board[r + 1, c]:
                    transitions += 1

        return transitions

    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes (empty cells with filled cells above them)."""
        holes = 0
        rows, cols = board.shape

        for c in range(cols):
            block_found = False
            for r in range(rows):
                if board[r, c] == 1:
                    block_found = True
                elif block_found and board[r, c] == 0:
                    holes += 1

        return holes

    def _count_cumulative_wells(self, board: np.ndarray) -> int:
        """Count cumulative wells (consecutive empty cells in columns)."""
        wells = 0
        rows, cols = board.shape

        for c in range(cols):
            well_depth = 0
            for r in range(rows):
                if board[r, c] == 0:
                    # Check if this is a well (surrounded by blocks or boundaries)
                    left_blocked = (c == 0) or (board[r, c - 1] == 1)
                    right_blocked = (c == cols - 1) or (board[r, c + 1] == 1)

                    if left_blocked and right_blocked:
                        well_depth += 1
                    else:
                        # Accumulate well depth
                        wells += well_depth * (well_depth + 1) // 2
                        well_depth = 0
                else:
                    # Accumulate well depth when hitting a block
                    wells += well_depth * (well_depth + 1) // 2
                    well_depth = 0

            # Accumulate remaining well depth
            wells += well_depth * (well_depth + 1) // 2

        return wells


class LinearTetrisPolicy:
    """
    Linear policy for Tetris as used in Szita & Lőrincz (2006).
    Uses linear combination of features to evaluate board positions.
    """

    def __init__(self, num_features: int = 6):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.feature_extractor = TetrisFeatureExtractor()

    def set_weights(self, weights: np.ndarray):
        """Set the linear weights for the policy."""
        assert len(weights) == self.num_features, (
            f"Expected {self.num_features} weights, got {len(weights)}"
        )
        self.weights = weights.copy()

    def evaluate_position(self, features: np.ndarray) -> float:
        """Evaluate a board position using linear combination of features."""
        return np.dot(self.weights, features)

    def select_action(self, board: np.ndarray, possible_actions: List[Tuple]) -> int:
        """
        Select the best action based on the resulting board evaluation.

        Args:
            board: Current board state
            possible_actions: List of (board_after_action, piece_info) tuples

        Returns:
            Index of the best action
        """
        best_score = float("-inf")
        best_action = 0

        for i, (resulting_board, piece_info) in enumerate(possible_actions):
            features = self.feature_extractor.extract_features(
                resulting_board, piece_info
            )
            score = self.evaluate_position(features)

            if score > best_score:
                best_score = score
                best_action = i

        return best_action


class NoisyCrossEntropyMethod:
    """
    Implementation of the Noisy Cross-Entropy Method for optimizing Tetris policy weights.
    Based on Szita & Lőrincz (2006) "Learning Tetris using the noisy cross-entropy method".
    """

    def __init__(
        self,
        num_features: int = 6,
        population_size: int = 100,
        elite_ratio: float = 0.1,
        noise_std: float = 1.0,
        initial_std: float = 1.0,
    ):
        """
        Initialize the Noisy Cross-Entropy Method.

        Args:
            num_features: Number of features (6 in original paper)
            population_size: Size of the population per generation
            elite_ratio: Fraction of population to select as elite
            noise_std: Standard deviation of noise added to elite samples
            initial_std: Initial standard deviation for weight sampling
        """
        self.num_features = num_features
        self.population_size = population_size
        self.num_elite = max(1, int(population_size * elite_ratio))
        self.noise_std = noise_std

        # Initialize distribution parameters
        self.mean = np.zeros(num_features)
        self.std = np.full(num_features, initial_std)

        # Track evolution
        self.generation = 0
        self.best_weights = None
        self.best_score = float("-inf")
        self.score_history = []

    def sample_population(self) -> np.ndarray:
        """Sample a population of weight vectors from current distribution."""
        samples = np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=(self.population_size, self.num_features),
        )
        return samples

    def add_noise_to_elite(self, elite_weights: np.ndarray) -> np.ndarray:
        """Add noise to elite samples to prevent premature convergence."""
        noise = np.random.normal(0, self.noise_std, elite_weights.shape)
        return elite_weights + noise

    def update_distribution(self, population: np.ndarray, scores: np.ndarray):
        """Update the sampling distribution based on elite samples."""
        # Select elite samples
        # elite_indices = np.argsort(scores)[-self.num_elite :]
        # elite_weights = population[elite_indices]

        # Instead of "elite ratio", CEM uses a performance threshold
        # Select samples above threshold γ (gamma)
        threshold = np.percentile(scores, 100 * (1 - self.num_elite / self.population_size))  # rho = fraction to keep
        selected_samples = population[scores >= threshold]

        # Add noise to elite samples (key innovation of the paper)
        noise = np.random.normal(0, self.noise_std, selected_samples.shape)
        noisy_samples = selected_samples + noise  # Add noise!

        # Then update parameters
        new_mean = np.mean(noisy_samples, axis=0)
        new_std = np.std(noisy_samples, axis=0)

        # Update distribution parameters
        self.mean = new_mean
        self.std = new_std

        # Prevent std from becoming too small
        self.std = np.maximum(self.std, 0.01)

        # Track best solution
        best_idx = np.argmax(scores)
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx]
            self.best_weights = population[best_idx].copy()

        self.score_history.append(scores[best_idx])
        self.generation += 1

    def get_best_policy(self) -> LinearTetrisPolicy:
        """Get the best policy found so far."""
        policy = LinearTetrisPolicy(self.num_features)
        if self.best_weights is not None:
            policy.set_weights(self.best_weights)
        return policy

    def get_stats(self) -> dict:
        """Get statistics about the optimization process."""
        return {
            "generation": self.generation,
            "best_score": self.best_score,
            "current_mean": self.mean.copy(),
            "current_std": self.std.copy(),
            "score_history": self.score_history.copy(),
        }
