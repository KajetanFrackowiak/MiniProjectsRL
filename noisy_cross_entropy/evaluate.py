import gymnasium as gym
import numpy as np
import argparse
import imageio
import os
from models import CEMPolicy

argparse = argparse.ArgumentParser()
argparse.add_argument("--env", type=str, default="CartPole-v1")
argparse.add_argument("--model", type=str, default="CartPole-v1_best_policy.npy")
argparse.add_argument("--num_episodes", type=int, default=5)
argparse.add_argument("--render", action="store_true")

args = argparse.parse_args()
env = gym.make("CartPole-v1", render_mode="rgb_array" if args.render else None)

def evaluate(env, model_path, num_episodes=5, render=False):
    weights = np.load(model_path)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    print(f"Observation Dimension: {obs_dim}, Action Dimension: {act_dim}")
    policy = CEMPolicy(obs_dim, act_dim, discrete=isinstance(env.action_space, gym.spaces.Discrete))
    print(f"Policy created with weights shape: {weights.shape}, expected: ({obs_dim * act_dim},)")
    policy.set_weights(weights)
    print("Starting evaluation...")
    total_reward = 0
    if render:
        frames = []
    for episode in range(num_episodes):
        print(f"Resetting environment for episode {episode + 1}")
        obs, _ = env.reset()
        print(f"Episode {episode + 1} starting with observation: {obs}")
        done = False
        episode_reward = 0
        print(f"Starting Episode {episode + 1}")
        while not done:
            print(f"Current observation: {obs}")
            if render:
                frames.append(env.render())
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            print(f"Step: {obs}, Action: {action}, Reward: {reward}")

        total_reward += episode_reward
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    avg_reward = total_reward / num_episodes
    if render:
        os.makedirs("videos", exist_ok=True)
    imageio.mimsave("videos/evaluation.mp4", frames, fps=30) if render else None
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    evaluate(env, args.model, args.num_episodes, args.render)
    env.close()