import gymnasium as gym
import numpy as np
import argparse
import imageio
import os
from CEM import CEMPolicy

argparse = argparse.ArgumentParser()
argparse.add_argument("--env", type=str, default="CartPole-v1")
argparse.add_argument("--model", type=str, required=True)
argparse.add_argument(
    "--method",
    type=str,
    required=True,
    choices=["Noisy_CrossEntropy", "CBMPI"],
)
argparse.add_argument("--num_episodes", type=int, default=5)
argparse.add_argument("--rgb_array", action="store_true", default=False)

args = argparse.parse_args()
env = gym.make(args.env, render_mode="rgb_array" if args.rgb_array else None)


def evaluate(
    env, model_path, method="Noisy_CrossEntropy", num_episodes=5, rgb_array=False
):
    obs_dim = env.observation_space.shape[0]
    act_dim = (
        env.action_space.n
        if isinstance(env.action_space, gym.spaces.Discrete)
        else env.action_space.shape[0]
    )

    if method == "Noisy_CrossEntropy":
        weights = np.load(model_path)
        policy = CEMPolicy(
            obs_dim, act_dim, discrete=isinstance(env.action_space, gym.spaces.Discrete)
        )
        policy.set_weights(weights)
    

    total_reward = 0
    if rgb_array:
        frames = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if rgb_array:
                frames.append(env.render())
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    avg_reward = total_reward / num_episodes
    if rgb_array:
        filename = args.model
        base_name = os.path.splitext(os.path.basename(filename))[0]
        os.makedirs("videos", exist_ok=True)
        imageio.mimsave(f"videos/{base_name}.mp4", frames, fps=30)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")


if __name__ == "__main__":
    evaluate(env, args.model, args.method, args.num_episodes, args.rgb_array)
    env.close()
