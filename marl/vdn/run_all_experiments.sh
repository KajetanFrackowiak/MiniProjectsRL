#!/bin/bash

# Script to run all 3 environments 5 times (5 seeds)
# Results are saved to results/seed_0, results/seed_1, ..., results/seed_4

# Create results directory if it doesn't exist
mkdir -p results

# Define environments
ENVS=(
    "simple_spread_v3"
    "simple_speaker_listener_v4"
    "simple_adversary_v3"
)

# Number of seeds
NUM_SEEDS=5

# Run training for each seed
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "Seed $seed"
    
    # Create seed directory
    SEED_DIR="results/seed_${seed}"
    mkdir -p "$SEED_DIR"
    
    # Run each environment for this seed
    for env in "${ENVS[@]}"; do
        echo "  Training on $env..."
        
        # Run the training script
        python main.py --env "$env" > /dev/null 2>&1
        
        # Move training_stats to seed directory
        if [ -d "training_stats" ]; then
            mv training_stats "$SEED_DIR/training_stats"
        fi
    done
done

echo "All training completed!"
echo "Generating plots..."

# Generate the combined plot
python plot_results.py --results_dir results

echo "Done! Check results_all_envs.png"
