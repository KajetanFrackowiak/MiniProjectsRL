import yaml
import os
import secrets
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import torch
import torch.optim as optim
from stable_baselines3 import PPO

from student import StudentPolicy
from training import Trainer
from expert import train_expert_model

def load_hyperparameters():
    with open("hyperparameters.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="DAGGER Imitation Learning")
    parser.add_argument('--env', type=str, default='MountainCar-v0', help='Gym environment name')
    parser.add_argument("--expert_path", type=str, default="", help="Path to the expert model")
    args = parser.parse_args()

    config = load_hyperparameters()

    
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    seed = config.get('seed', secrets.randbits(32))
    env.reset(seed=seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.expert_path):
        print("Training expert model...")
        train_expert_model(
            env=env,
            env_name=args.env,
            path=args.expert_path,
            seed=seed,
            epochs=config['expert_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['expert_learning_rate'],
            gamma=config['expert_gamma'],
            n_steps=config['expert_n_steps'],
            total_timesteps=config['expert_total_timesteps'],
            device=device,
        )
    else:
        print("Expert model already exists. Skipping training.")

    print("Loading expert model...")
    expert_path = args.expert_path if args.expert_path else f"experts/expert_env_{args.env}_epochs_{config['expert_epochs']}_seed_{seed}.zip"
    expert = PPO.load(expert_path, env=env, device=device)

    student = StudentPolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=config['student_learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Training student policy using DAGGER...")
    trainer = Trainer(env, args.env, seed, expert, student, optimizer, loss_fn, device)
    agg_states, agg_actions, losses = trainer.run(
        iterations=config['student_iterations'],
        rollouts_per_iter=config['student_rollouts_per_iteration'],
        p=config['student_beta_decay']
    )

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Student Policy Training Loss')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()