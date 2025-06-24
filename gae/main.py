import torch
import numpy as np
import os
import wandb
import argparse
import gymnasium as gym
from mujoco_env import MuJoCoCartPoleEnv
from models import LinearPolicy, ValueFunction, MLPPolicy, MLPValueFunction, CNNDiscretePolicy, CNNValueFunction
from gae import trpo_update, compute_gae, update_value_function, ppo_update
from utils import preprocess_frame
class FrameStacker:
    def __init__(self, env, k, preprocess_fn=preprocess_frame):
        self.env = env
        #TODO

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
        if args.env == 2 or args.env == 3:
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
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=int,
                        choices=[1,2,3,4,5,6],
                        required=True,
                        help="Environments to use:"
                        "1: PongNoFrameskip-v4"
                        "2: CartPole-v1"
                        "3: Pendulum-v1" 
                        "4: Ant-v4"
                        "5: Humanoid-v5" 
                        "6: CartPole_own")
    
    parser.add_argument("--trust_region",
                        type=int,
                        choices=[1,2],
                        required=True,
                        help="Choose algorithm to trust regions:"
                        "1: TRPO"
                        "2: PPO")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--lambda_",
                        type=float,
                        default=1,
                        help="Lambda parameter")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.95,
                        help="Gamma parameter")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.env == 1:
        env = gym.make("PongNoFrameskip-v4")
        env_name = "pongNoFrameskip-v4"
    elif args.env == 2:
        env = gym.make("CartPole-v1")
        env_name = "cartpole-v1"
    elif args.env == 3:
        env = gym.make("Pendulum-v1")
        env_name = "pendulum-v1"
    elif args.env == 4:
        env = gym.make("Ant-v4")
        env_name = "ant-v4"
    elif args.env == 5:
        env = gym.make("Humanoid-v5")
        env_name = "humanoid-v5"
    elif args.env == 6:
        env = MuJoCoCartPoleEnv()
        env_name = "cartpole_own"

    if args.trust_region == 1:
        trust_region_name = "trpo"
    elif args.trust_region == 2:
        trust_region_name = "ppo"

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    wandb.init(
        project=f"gae",
        name=f"{trust_region_name}_{env_name}_lambda_{args.lambda_}_gamma_{args.gamma}_seed_{args.seed}",
        config={
        "gamma": args.gamma,
        "max_timesteps_per_batch": 4000 if args.env == 1 else 10_000,
        "max_iters": 50 if args.env == 1 else 500,
        "gae_lambdas": args.lambda_,
        "value_lr": 1e-3,
        "policy_lr": 1e-3,
        "max_kl": 1e-2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

    })

    config = wandb.config
    device = torch.device(config.device)

    if args.env == 1:

        policy = CNNDiscretePolicy(obs_dim, act_dim).to(device)
        value_fn = CNNValueFunction(obs_dim).to(device)
    elif args.env == 2:

    elif args.env in [3,6]:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        policy = LinearPolicy(obs_dim, act_dim).to(device)
        value_fn = ValueFunction(obs_dim).to(device)
    elif args.env in [4,5]:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        policy = MLPPolicy(obs_dim, act_dim).to(device)
        value_fn = MLPValueFunction(obs_dim).to(device)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=config.value_lr)

    best_avg_return = -float("inf")

    for iteration in range(config.max_iters):
        data = collect_episodes(args, env, policy, value_fn, config.max_timesteps_per_batch, config.gamma, device)

        advantages = compute_gae(
            data['rewards'], data['values'], data['dones'], config.gamma, args.lambda_
        )
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + torch.tensor(data['values'][:-1], dtype=torch.float32).to(device)

        # Move tensors to device
        observations = data['observations'].to(device)
        actions = data['actions'].to(device)
        old_log_probs = data['log_probs'].clone().detach().to(device)

        # Policy update
        if args.trust_region == 1:
            trpo_update(policy, observations, actions, advantages, old_log_probs, max_kl=config.max_kl)
        elif args.trust_region == 2:
            ppo_update(policy, value_fn, policy_optimizer, observations,
                       actions, advantages, returns, old_log_probs)

        # Value function update
        update_value_function(value_fn, value_optimizer, observations, returns)

        avg_return = np.mean(data['episode_returns']) if data['episode_returns'] else 0
        wandb.log({
            "lambda": args.lambda_,
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
            }, os.path.join(checkpoint_dir, f"{trust_region_name}_avg_{avg_return:.2f}_lambda_{args.lambda_:.2f}_gamma_{args.gamma:.2f}_{env_name}.pth"))
            print(f"Saved new best mode at iteration {iteration} with avg return {avg_return:.2f}")
if __name__ == "__main__":
    main()
