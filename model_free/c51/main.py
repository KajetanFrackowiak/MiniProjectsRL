import argparse
import yaml
import os
import secrets
import numpy as np
import ale_py
import gymnasium as gym
import wandb
import torch
from tqdm import tqdm

from agent import C51Agent
from utils import FrameStacker


def load_hyperparameters(env_id):
    try:
        with open("hyperparameters.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            "hyperparameters.yaml file not found. Please create it with the necessary configurations."
        )
    return config.get(env_id, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        choices=[1, 2, 3],
        help="1: PongNoFrameskip-v4: BreakoutNoFrameskip-v4: SeaquentNoFrameskip-v4",
    )
    parser.add_argument("--seed", type=int, default=secrets.randbelow(2**32))
    args = parser.parse_args()

    env_nams = {
        1: "PongNoFrameskip-v4",
        2: "BreakoutNoFrameskip-v4",
        3: "SeaquestNoFrameskip-v4",
    }
    env_name = env_nams[args.env]
    env = gym.make(env_name)


    np.random.seed(args.seed)
    env.reset(seed=args.seed)
    env.observation_space.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_hyperparameters(env_name)

    wandb.init(
            project="c51",
            name=f"{env_name}_seed_{args.seed}",
            config={
                "env_name": env_name,
                "seed": args.seed,
                "frame_stack": config.get("frame_stack", 4),
                "num_atoms": config.get("num_atoms", 51),
                "Vmin": config.get("Vmin", -10),
                "Vmax": config.get("Vmax", 10),
                "lr": config.get("lr", 1e-4),
                "max_iters": config.get("max_iters", 10000),
            },
        )
    
    env = FrameStacker(env, k=config["frame_stack"])
    
    act_dim = env.get_act_dim()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print()

    print(f"Using device: {device}, Seed: {args.seed}")

    os.makedirs("checkpoints", exist_ok=True)

    agent = C51Agent(
        stacked_frames=config["frame_stack"],
        act_dim=act_dim,
        num_atoms=config["num_atoms"],
        Vmin=config["Vmin"],
        Vmax=config["Vmax"],
        lr=config["lr"],
        device=device,
    )

    best_avg_return = -float("inf")
    all_returns = []

    for iteration in tqdm(range(config["max_iters"]), desc="Training Progress"):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        train_steps = 0
        while not done:
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = agent.get_action(obs_tensor)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(obs, action.item(), reward, next_obs, done)
        
            obs = next_obs
            agent.train_step()
            train_steps += 1
        all_returns.append(total_reward)
        avg_return = np.mean(all_returns[-100:]) if len(all_returns) >= 100 else total_reward
        wandb.log(
            {
                "iteration": iteration + 1,
                "avg_return": avg_return,
                "train_steps": train_steps,
            }
        )
        tqdm.write(f"Iteration {iteration + 1}, Average Return: {avg_return:.2f}")
        if avg_return > best_avg_return and iteration % 100 == 0:
            best_avg_return = avg_return
            torch.save(agent.state_dict(), f"checkpoints/{env_name}_best.pth")
            tqdm.write(f"New best model saved with average return: {avg_return:.2f}")

if __name__ == "__main__":
    main()