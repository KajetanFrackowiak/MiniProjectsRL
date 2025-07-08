import cv2
import os
import torch
import ale_py
import numpy as np
import yaml
import argparse
import secrets
import gymnasium as gym
from collections import deque
from torch import optim
from models import WorldModelStochastic, Policy
from utils import ReplayBuffer, ppo_update, FrameStacker, preprocess_frame, collect_episodes

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SimulatedEnv:
    """
    Simulated environment using the world model.
    """
    def __init__(self, world_model, device):
        self.world_model = world_model
        self.device = device
        self.reset_state = None
        self.current_obs = None
        self.done = False

    def reset(self, obs):
        self.current_obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.done = False
        return self.current_obs

    def step(self, action):
        # action: int or one-hot
        if isinstance(action, int):
            action = np.eye(self.world_model.action_emb.in_features)[action]
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_frame_logits, reward, _, _ = self.world_model(self.current_obs, action, mode='inference')
            # Per-pixel softmax
            probs = torch.softmax(next_frame_logits, dim=2)
            # Sample next frame
            next_frame = torch.argmax(probs, dim=2).float() / 255.0  # [B, 3, H, W]
            reward = reward.item()
        self.current_obs = next_frame
        # Simulated env does not know true done, so always False
        return next_frame.squeeze(0).cpu().numpy(), reward, self.done, {}

# Scheduled sampling utility
def scheduled_sample(real, pred, prob):
    mask = np.random.rand(*real.shape) < prob
    return np.where(mask, pred, real)

def train_world_model(world_model, buffer, optimizer, device, batch_size=64, steps=10000, scheduled_sampling=False, ss_prob=0.0, loss_clip=0.03, bit_dropout=0.2, env_name=""):
    world_model.train()
    for step in range(steps):
        if len(buffer) < batch_size:
            continue
        obs, action, reward, next_obs, done = buffer.sample(batch_size)
        obs = obs.to(device)
        next_obs = next_obs.to(device)
        action = torch.nn.functional.one_hot(action, num_classes=world_model.action_emb.in_features).float().to(device)
        reward = reward.to(device)
        # Scheduled sampling: replace some obs with model prediction
        if scheduled_sampling and step > steps // 10:
            with torch.no_grad():
                pred_next_logits, _, _, _ = world_model(obs, action, mode='inference')
                pred_next = torch.argmax(torch.softmax(pred_next_logits, dim=2), dim=2).float() / 255.0
            ss_mask = (torch.rand_like(obs) < ss_prob).float()
            obs = ss_mask * pred_next + (1 - ss_mask) * obs
        # Forward pass
        next_frame_logits, pred_reward, bits, bit_probs = world_model(obs, action, next_obs, mode='train')
        # Per-pixel softmax loss
        # Use only the last channel of next_obs as the target (the next grayscale frame)
        target = (next_obs[:, -1:, :, :] * 255).long()  # [B, 1, 84, 84]
        logits = next_frame_logits.permute(0,1,3,4,2).reshape(-1, 256) # [B*1*84*84, 256]
        target = target.reshape(-1)  # [B*1*84*84]
        # Visualize target and prediction for the first sample in the batch every 1000 steps
        if step % 1000 == 0:
            # Target image (ground truth)
            target_img = target[:84*84].reshape(84, 84).detach().cpu().numpy().astype(np.uint8)
            # Predicted image (argmax over logits)
            pred_img = torch.argmax(next_frame_logits[0, 0], dim=0).detach().cpu().numpy().astype(np.uint8)
            # Stack side by side for comparison
            comparison = np.hstack([target_img, pred_img])
            cv2.imwrite(f"{env_name}_train_step_{step}.png", comparison)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
        # Reward loss (MSE)
        reward_loss = torch.nn.functional.mse_loss(pred_reward.squeeze(), reward.squeeze())
        # Remove extra bit dropout - already handled in model
        total_loss = loss + reward_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"World model step {step}, loss: {total_loss.item():.4f}")
            # Save checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': world_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item()
            }
            torch.save(checkpoint, os.path.join('checkpoints', f'{env_name}_step_{step}.pth'))

def collect_real_env_data(env, policy, buffer, num_steps):
    obs, _ = env.reset()
    for _ in range(num_steps):
        obs_proc = obs
        action = policy(obs_proc)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

def collect_simulated_rollouts(sim_env, policy, buffer, rollout_length=50, num_envs=16):
    rollouts = []
    for _ in range(num_envs):
        # Sample a real state from buffer
        idx = np.random.randint(0, len(buffer))
        obs, _, _, _, _ = buffer.buffer[idx]
        obs = obs.cpu().numpy()
        sim_env.reset(obs)
        for _ in range(rollout_length):
            action = policy(obs)
            next_obs, reward, termianted, truncated, _ = sim_env.step(action)
            done = termianted or truncated
            rollouts.append((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                break
    # Convert to tensors for PPO
    obs, action, reward, next_obs, done = zip(*rollouts)
    return (torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(action)),
            torch.tensor(np.array(reward), dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32))

def random_policy(obs):
    return env.action_space.sample()

def simple_main(env, real_policy, world_model, policy, policy_optimizer, device, config, env_name):


    buffer = ReplayBuffer(config['buffer_capacity'])
    # 1. Initial random data collection
    collect_real_env_data(env, real_policy, buffer, config['init_steps'])
    for iteration in range(config['num_iterations']):
        print(f"=== SimPLe Iteration {iteration} ===")
        # 2. Train world model
        wm_optimizer = optim.Adam(world_model.parameters(), lr=config['wm_lr'])
        train_world_model(world_model, buffer, wm_optimizer, device,
                         batch_size=config['wm_batch_size'],
                         steps=config['wm_steps'],
                         scheduled_sampling=True,
                         ss_prob=config['ss_prob'],
                         loss_clip=config['loss_clip'],
                         bit_dropout=config['bit_dropout'],
                         env_name=env_name)
        # 3. Train policy in simulated env
        sim_env = SimulatedEnv(world_model, device)
        for ppo_epoch in range(config['ppo_epochs']):
            rollouts = collect_simulated_rollouts(sim_env, policy, buffer,
                                                  rollout_length=config['rollout_length'],
                                                  num_envs=config['num_envs'])
            ppo_update(policy, policy_optimizer, *rollouts, config)
        # 4. Collect more real data
        collect_real_env_data(env, policy, buffer, config['real_steps_per_iter'])

        # 5. Evaluate policy in real environment every few iterations
        if (iteration + 1) % 2 == 0 or iteration == config['num_iterations'] - 1:
            mean_reward = evaluate_policy_mean_reward(env, policy, device, episodes=10)
            print(f"[Eval] Iteration {iteration+1}: Mean reward over last 10 episodes: {mean_reward}")

def evaluate_policy_mean_reward(env, policy, device, episodes=10):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_tensor)
                action = dist.sample().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        rewards.append(total_reward)
    mean_last = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    print(f"Mean reward over last {min(100, len(rewards))} episodes: {mean_last}")
    return mean_last


if __name__ == "__main__":
    checkpoints_dir = "./checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Environemnts to use:" \
        "1: PongNoFrameskip-v4"
        "2: BreakoutNoFrameskip-v4"
        "3: SeaquestNoFrameskip-v4"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=secrets.randbelow(2**32)
    )

    args = parser.parse_args()
    if args.env == 1:
        env_name = "PongNoFrameskip-v4"
    elif args.env == 2:
        env_name = "BreakoutNoFrameskip-v4"
    else:
        env_name = "SeaquestNoFrameskip-v4"
    

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    env = gym.make(env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    env = FrameStacker(env, k=4, preprocess_fn=preprocess_frame)

    print(f"Using device: {device}")
    config = load_config("hyperparameters.yaml")

    
    world_model = WorldModelStochastic(
        input_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        input_height=env.observation_space.shape[1],
        input_width=env.observation_space.shape[2],
        num_bits=config.get("num_bits", 32),
    ).to(device)
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=config['policy_lr'])


    simple_main(env, random_policy, world_model, policy, policy_optimizer, device, config, env_name)
