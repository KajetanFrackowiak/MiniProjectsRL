import gymnasium as gym
import yaml
import numpy as np
import argparse
from models import CEMTrainer

def load_hyperparameters(env_name):
    with open('hyperparameters.yaml', 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams.get(env_name, {})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    
    args = parser.parse_args()
    env_name = args.env
    hyperparams = load_hyperparameters(env_name)
    trainer = CEMTrainer(env_name, 
                         population_size=hyperparams.get('population_size', 30),
                         elite_ratio=hyperparams.get('elite_ratio', 0.2))
    best_policy = trainer.train(num_generations=hyperparams.get('num_generations', 100),
                                num_episodes=hyperparams.get('num_episodes', 5))
    
    print("Saving traiend policies...")
    print(f"Best policy weights shape: {best_policy.get_weights_flat().shape}")
    np.save(f"{env_name}_best_policy.npy", best_policy.get_weights_flat())


if __name__ == "__main__":
    main()