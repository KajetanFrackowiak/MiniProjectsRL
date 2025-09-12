import os
import yaml
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import gymnasium as gym
import argparse
import secrets
from tqdm import tqdm

def load_hyperparameters():
    with open("hyperparameters.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_expert_model(
    env,
    env_name,
    seed,
    epochs,
    batch_size,
    learning_rate,
    gamma,
    n_steps,
    total_timesteps,
    device,
):
    env = Monitor(env, filename="training_log.csv")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        device=device,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("experts", exist_ok=True)
    model.save(f"experts/expert_env_{env_name}_epochs_{epochs}_seed_{seed}.zip")

    state, _ = env.reset()
    for i in range(epochs):
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            state, _ = env.reset()

    env.close()


def infer_expert_model(env, model_path, device="cuda"):
    os.makedirs("logs", exist_ok=True)
    env = Monitor(env, filename="logs/inference_log.")
    model = PPO.load(model_path, device=device)
    try:
        for i in tqdm(range(1000)):
            state, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                env.render()
                done = terminated or truncated

            print(f"Inference step {i + 1}/1000 completed.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0', help='Gym environment name')
    parser.add_argument("--expert_path", type=str, default="", help="Path to the expert model")

    args = parser.parse_args()
    config = load_hyperparameters()

    
    if args.expert_path != "":
        env = gym.make("MountainCar-v0", render_mode="human")
        infer_expert_model(
            env,
            model_path=args.expert_path,
        )
    else:
        env = gym.make("MountainCar-v0")

        train_expert_model(
            env,
            env_name="MountainCar-v0",
            seed= secrets.randbits(32),
            epochs=config["expert_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["expert_learning_rate"],
            gamma=config["expert_gamma"],
            n_steps=config["expert_n_steps"],
            total_timesteps=config["expert_total_timesteps"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

