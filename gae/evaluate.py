import torch
import gymnasium as gym
from models import LinearPolicy, ValueFunction, MLPValueFunction, MLPPolicy
from mujoco_env import MuJoCoCartPoleEnv
import argparse

def load_model(args, filepath, obs_dim, act_dim, device):
    checkpoint = torch.load(filepath, map_location=device)

    if args.env == 1:
        policy = LinearPolicy(obs_dim, act_dim).to(device)
        value_fn = ValueFunction(obs_dim).to(device)
    else:
        policy = MLPPolicy(obs_dim, act_dim).to(device)
        value_fn = MLPValueFunction(obs_dim).to(device)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    value_fn.load_state_dict(checkpoint["value_fn_state_dict"])

    policy.eval()
    value_fn.eval()

    return policy, value_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--env",
                        type=int,
                        choices=[1,2,3],
                        required=True,
                        help="Environments to use:"
                        "1: CartPole"
                        "2: Ant-v4"
                        "3: Humanoid-v5")
args = parser.parse_args()
if args.env == 1:
    env = MuJoCoCartPoleEnv()
elif args.env == 2:
    env = gym.make("Ant-v4", render_mode="human")
elif args.env == 3:
    env = gym.make("Humanoid-v5", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy, value_fn = load_model(args, "model_avg_448.21_lambda_1.00_humanoid.pth", obs_dim, act_dim, device)
episode_rewards = []

while True:  # Infinite episodes
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        dist = policy(obs_tensor)
        if args.env == 1:
            action = dist.sample().detach().cpu().numpy()
        else:
            action = dist.sample().detach().cpu().numpy().reshape(-1)
        observation, reward, truncated, terminated, info = env.step(action)
        done = truncated or terminated
        env.render()
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode reward: {total_reward}")

    if len(episode_rewards) >= 100:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"Average reward over last 100 episodes: {avg_reward}")
