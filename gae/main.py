import torch
import numpy as np
import wandb
import argparse
import gymnasium as gym
from mujoco_env import MuJoCoCartPoleEnv
from models import LinearPolicy, ValueFunction, MLPPolicy, MLPValueFunction # or your model names
from trpo import trpo_update, compute_gae, update_value_function

def collect_episodes(args, env, policy, value_fn, target_timesteps, gamma, device):
    observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    episode_returns = []
    episode_return = 0
    episode_length = 0
    timesteps_collected = 0

    obs = env.reset()[0]

    while timesteps_collected < target_timesteps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        dist = policy(obs_tensor)
        action = dist.sample()
        if args.env == 2:
            action = action.squeeze(0)
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = value_fn(obs_tensor).item()

        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated

        observations.append(obs)
        actions.append(action.cpu().numpy())
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob.item())

        episode_return += reward
        episode_length += 1
        timesteps_collected += 1

        obs = observation
        if done:
            episode_returns.append(episode_return)
            episode_return = 0
            episode_length = 0
            obs = env.reset()[0]

    # Append value for last step for bootstrapping
    values.append(value_fn(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)).item())

    return {
        'observations': torch.tensor(np.array(observations), dtype=torch.float32),
        'actions': torch.tensor(np.array(actions).squeeze(), dtype=torch.float32),
        'rewards': rewards,
        'dones': dones,
        'values': values,
        'log_probs': torch.tensor(log_probs, dtype=torch.float32),
        'episode_returns': episode_returns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=int,
                        choices=[1,2],
                        help="Environments to use:"
                        "1: CartPole"
                        "2: Ant-v4")
    args = parser.parse_args()

    wandb.init(project=f"trpo_{args.env}", config={
        "gamma": 0.95,
        "max_timesteps_per_batch": 4000 if args.env == 1 else 10_000,
        "max_iters": 50,
        "gae_lambdas": [1.0, 0.0, 0.99],
        "value_lr": 1e-3,
        "max_kl": 1e-2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })

    config = wandb.config

    if args.env == 1:
        env = MuJoCoCartPoleEnv()
    else:
        env = gym.make("Ant-v4")
    env_name = "cartpole" if args.env == 1 else "ant"
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device(config.device)

    for lam in config.gae_lambdas:
        wandb.log({"lambda": lam})
        print(f"Training with GAE lambda = {lam:.2f}")

        if args.env == 1:
            policy = LinearPolicy(obs_dim, act_dim).to(device)
            value_fn = ValueFunction(obs_dim).to(device)
        else:
            policy = MLPPolicy(obs_dim, act_dim).to(device)
            value_fn = MLPValueFunction(obs_dim).to(device)
        value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=config.value_lr)
        best_avg_return = -float("inf")

        for iteration in range(config.max_iters):
            data = collect_episodes(args, env, policy, value_fn, config.max_timesteps_per_batch, config.gamma, device)

            advantages = compute_gae(
                data['rewards'], data['values'], data['dones'], config.gamma, lam
            )
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            returns = advantages + torch.tensor(data['values'][:-1], dtype=torch.float32).to(device)

            # Move tensors to device
            observations = data['observations'].to(device)
            actions = data['actions'].to(device)
            old_log_probs = data['log_probs'].clone().detach().to(device)

            # Policy update
            trpo_update(policy, observations, actions, advantages, old_log_probs, max_kl=config.max_kl)

            # Value function update
            update_value_function(value_fn, value_optimizer, observations, returns)

            avg_return = np.mean(data['episode_returns']) if data['episode_returns'] else 0
            wandb.log({
                "lambda": lam,
                "iteration": iteration,
                "avg_return": avg_return,
            })

            print(f"Iter {iteration} | Avg Return: {avg_return:.2f} | Episodes: {len(data['episode_returns'])}")

            if avg_return > best_avg_return:
                best_avg_return = avg_return
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "value_fn_state_dict": value_fn.state_dict(),
                    "optimizer_state_dict": value_optimizer.state_dict(),
                    "iteration": iteration,
                    "avg_return": avg_return,
                }, f"model_avg_{avg_return:.2f}_lambda_{lam:.2f}_{env_name}.pth")
            print(f"Saved new best mode at iteration {iteration} with avg return {avg_return:.2f} "
                  f"and lambda: {lam:.2f}")
if __name__ == "__main__":
    main()
