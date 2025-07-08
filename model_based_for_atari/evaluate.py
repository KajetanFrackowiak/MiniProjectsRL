import cv2
import torch
import numpy as np
import gymnasium as gym
import ale_py
import argparse

from models import WorldModelStochastic
from utils import FrameStacker, preprocess_frame


def visualize_world_model_vs_real(env, world_model, device, policy=None, steps=1000):
    obs, _ = env.reset()
    obs_stack = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    world_model.eval()
    for i in range(steps):
        # Choose action
        if policy is not None:
            action = policy(obs)
        else:
            action = env.action_space.sample()
        # Real env step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # World model prediction
        action_onehot = torch.nn.functional.one_hot(
            torch.tensor([action], device=device), num_classes=world_model.action_emb.in_features
        ).float()
        with torch.no_grad():
            pred_logits, _, _, _ = world_model(obs_stack, action_onehot, mode='inference')
            pred_frame = torch.argmax(torch.softmax(pred_logits, dim=2), dim=2).float() / 255.0  # [1, 1, H, W]
        # Prepare images for display
        real_img = (next_obs[-1] * 255).astype(np.uint8)  # last channel of stack
        pred_img = (pred_frame.squeeze().cpu().numpy() * 255).astype(np.uint8)
        scale = 1
        real_img = cv2.resize(real_img, (real_img.shape[1]*scale, real_img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        pred_img = cv2.resize(pred_img, (pred_img.shape[1]*scale, pred_img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        # print(f"Step {i}: real_img shape {real_img.shape}, dtype {real_img.dtype}; pred_img shape {pred_img.shape}, dtype {pred_img.dtype}")
        both = np.hstack([real_img, pred_img])
        # print(f"Displaying frame {i}, both shape: {both.shape}")
        cv2.imshow('Real (left) vs World Model Prediction (right)', both)
        key = cv2.waitKey(10) # wait for 10 ms
        if key == ord("q"):
            # print("Quitting visualization loop.")
            break
        # Update obs_stack for world model: drop oldest, append predicted
        obs_stack = torch.cat([obs_stack[:, 1:], pred_frame], dim=1)
        obs = next_obs
        if done:
            obs, _ = env.reset()
            obs_stack = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    cv2.destroyAllWindows()

def evaluate_policy_mean_reward(env, policy, device, episodes=100):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if policy is not None:
                with torch.no_grad():
                    dist = policy(obs_tensor)
                    action = dist.sample().item()
            else:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        rewards.append(total_reward)
        if len(rewards) > 100:
            rewards = rewards[-100:]
        mean_last_100 = np.mean(rewards[-100:])
        print(f"Episode {ep+1}, Reward: {total_reward}, Mean of last 100: {mean_last_100}")
    print(f"Final mean reward over last 100 episodes: {np.mean(rewards[-100:])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=int, 
                        choices=[1,2,3],
                        help="Environments to choose:"
                        "1: PongNoFrameskip-v4"
                        "2: BreakoutNoFrameskip-v4"
                        "3: SeaquestNoFrameskip-v4")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    env_name = {1: "PongNoFrameskip-v4", 2: "BreakoutNoFrameskip-v4", 3: "SeaquestNoFrameskip-v4"}[args.env]
    env = gym.make(env_name)

    env = FrameStacker(env, k=4, preprocess_fn=preprocess_frame)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    world_model = WorldModelStochastic(env.observation_space.shape[0], env.action_space.n).to(device)
    world_model_policy = world_model.load_state_dict(checkpoint["model_state_dict"])
    world_model.eval()
    visualize_world_model_vs_real(env, world_model, device, policy=None, steps=1000)
    evaluate_policy_mean_reward(env, world_model, device, episodes=100)