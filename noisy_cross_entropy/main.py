import yaml
import numpy as np
import argparse
import secrets
import os
from CEM import CEMTrainer
from CBMPI import CBMPI


def load_hyperparameters(method, env_name):
    with open("hyperparameters.yaml", "r") as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams.get(method, {}).get(env_name, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument(
        "--decreasing_noise",
        action="store_true",
        default=True,
        help="Use decreasing noise schedule as in Szita & Lőrincz paper",
    )
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("-seed", type=int, default=secrets.randbelow(2**32))

    args = parser.parse_args()
    env_name = args.env
    hyperparams = load_hyperparameters(method=args.method, env_name=env_name)
    if args.method == "Noisy_CrossEntropy":
        print(f"Num generations: {hyperparams.get('num_generations', 1000)}")
        trainer = CEMTrainer(
            env_name,
            population_size=hyperparams.get("population_size", 30),
            elite_ratio=hyperparams.get("elite_ratio", 0.2),
            method=args.method,
            seed=args.seed,
            decreasing_noise=args.decreasing_noise,
        )
        avg_return, best_policy = trainer.train(
            num_generations=hyperparams.get("num_generations", 1000),
            num_episodes=hyperparams.get("num_episodes", 5),
        )
  
    print("Saving trained policies...")
    os.makedirs("models", exist_ok=True)
    if args.method == "Noisy_CrossEntropy":
        np.save(
            f"models/{args.method}_{env_name}_avg_{avg_return}_seed_{args.seed}.npy",
            best_policy.get_weights_flat(),
        )
   


if __name__ == "__main__":
    main()
