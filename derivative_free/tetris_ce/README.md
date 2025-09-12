# Learning Tetris Using the Noisy Cross-Entropy Method

This repository implements the approach described in the paper:

**"Learning Tetris using the noisy cross-entropy method"**  
*István Szita, András Lőrincz (2006)*  
Neural Computation, 18(12), 2936-2941

## Overview

The implementation provides:

1. **Tetris Feature Extractor**: Extracts the 6 handcrafted features used in the original paper
2. **Linear Tetris Policy**: Uses linear combination of features to evaluate board positions
3. **Noisy Cross-Entropy Method**: Optimization algorithm with noise injection to prevent premature convergence
4. **Atari Tetris Integration**: Wrapper for the Gymnasium Atari Tetris environment
5. **Simple Environment Demos**: Examples using CartPole and Pendulum

## Key Features from the Paper

The original paper uses 6 features to evaluate Tetris board states:

1. **Landing Height**: Height where the last piece landed
2. **Eroded Piece Cells**: Number of piece cells eliminated by line clears
3. **Row Transitions**: Horizontal transitions between filled/empty cells
4. **Column Transitions**: Vertical transitions between filled/empty cells  
5. **Holes**: Empty cells with filled cells above them
6. **Cumulative Wells**: Weighted sum of well depths

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Atari environments, you may also need:
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"
```

## Usage

### 1. Train on Atari Tetris

```bash
# Basic training
python train_tetris.py

# Custom parameters
python train_tetris.py --population-size 50 --generations 100 --elite-ratio 0.2

# All options:
python train_tetris.py --help
```

### 2. Test a Trained Policy

```bash
# Test saved weights
python train_tetris.py test tetris_weights.npy 10
```

### 3. Simple Environment Demos

```bash
# Train on CartPole and Pendulum
python simple_cem_demo.py
```

## Files

- **`models.py`**: Core implementation of feature extraction, policy, and CEM algorithm
- **`tetris_environment.py`**: Atari Tetris wrapper and training logic
- **`train_tetris.py`**: Main training script with command-line interface
- **`simple_cem_demo.py`**: Demo on CartPole and Pendulum environments
- **`requirements.txt`**: Python dependencies

## The Noisy Cross-Entropy Method

The key innovation in Szita & Lőrincz's approach is adding noise to elite samples:

```python
# Standard CEM selects elite samples
elite_weights = population[elite_indices]

# Noisy CEM adds noise to prevent premature convergence  
noise = np.random.normal(0, noise_std, elite_weights.shape)
noisy_elite = elite_weights + noise

# Update distribution based on noisy elite
mean = np.mean(noisy_elite, axis=0)
std = np.std(noisy_elite, axis=0)
```

This prevents the algorithm from converging too quickly to suboptimal solutions.

## Expected Performance

The original paper achieved:

- **35,000+ lines** cleared on average
- **Best run**: ~750,000 lines cleared
- Training typically converges within **100-200 generations**

Note: Performance depends heavily on:
- Quality of board state extraction from Atari visuals
- Accuracy of feature computation
- Hyperparameter tuning

## Hyperparameters

Key parameters for tuning:

- **Population Size**: 20-100 (larger = more stable, slower)
- **Elite Ratio**: 0.1-0.3 (fraction of best individuals to keep)
- **Noise Std**: 0.1-1.0 (higher = more exploration)
- **Games per Evaluation**: 3-10 (more games = more reliable but slower)

## Differences from Original Paper

This implementation makes some simplifications:

1. **Visual Processing**: The Atari environment requires extracting board state from pixels, which is challenging
2. **Action Space**: Maps continuous feature evaluation to discrete Atari actions
3. **Piece Information**: Some features require tracking individual pieces, which is simplified

For a complete implementation, you would need:
- More sophisticated visual processing to extract exact board state
- Full Tetris game logic to enumerate all possible piece placements
- Better mapping from feature evaluation to action selection

## Example Output

```
Generation  10: Best=  450.00, Mean=  125.50, Overall Best=  450.00
Generation  20: Best=  780.00, Mean=  234.20, Overall Best=  780.00
Generation  30: Best= 1250.00, Mean=  456.10, Overall Best= 1250.00

Final feature weights:
  landing_height      : -0.5123
  eroded_piece_cells  :  1.2456
  row_transitions     : -0.3789
  col_transitions     : -0.2345
  holes               : -2.1567
  cumulative_wells    : -0.4321
```

## References

1. Szita, I., & Lőrincz, A. (2006). Learning Tetris using the noisy cross-entropy method. *Neural computation*, 18(12), 2936-2941.

2. Rubinstein, R. Y., & Kroese, D. P. (2004). *The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation and machine learning*. Springer.

## License

This code is provided for educational and research purposes.
