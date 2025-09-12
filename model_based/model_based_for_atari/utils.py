import torch
import gymnasium as gym
import collections
import random
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

def collect_episodes(env, policy, target_timesteps, device):
    observations, actions, rewards, dones, log_probs = [], [], [], [], []
    episode_returns = []
    episode_return = 0
    timesteps_collected = 0

    obs, _ = env.reset()[0]

    while timesteps_collected < target_timesteps:
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).to(device)

        dist = policy(obs_tensor)
        action = dist.sample()

        # Convert action to numpy and ensure it is a scalar if needed
        action_for_env = (action.item() if action.dim() == 0 else action.squeeze().item())
        log_prob = dist.log_prob(action)
        while log_prob.dim() > 1:
            log_prob = log_prob.sum()
        
    observation, reward, terminated, truncated, _ = env.step(action_for_env)
    done = terminated or truncated
    
    observations.append(observation)
    actions.append(action.item() if action.dim() == 0 else action.squeeze().item())
    rewards.append(reward)
    dones.append(done)
    log_probs.append(log_prob.item())

    episode_return += reward
    timesteps_collected += 1

    obs = observation
    if done:
        episode_returns.append(episode_return)
        episode_return = 0 
        obs = env.reset()[0]
    
    return {
        "observations": torch.tensor(np.array(observations), dtype=torch.float32),
        "actions": torch.tensor(np.array(actions), dtype=torch.float32),
        "rewards": rewards,
        "dones": dones,
        "log_probs": torch.tensor(log_probs, dtype=torch.float32),
        "episode_returns": episode_returns,
    }


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32) 
        # [action] is wrapped in a list to ensure it is treated as a single action
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # *random.sample* is used to randomly select a batch of experiences from the buffer
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # squeeze() is used to remove any dimensions of size 1 from the tensors
        return (torch.stack(state), 
                torch.stack(action).squeeze(), 
                torch.stack(reward).squeeze(), 
                torch.stack(next_state), 
                torch.stack(done).squeeze())

    # This method is used to check if the buffer has enough samples to sample a batch
    def __len__(self):
        return len(self.buffer)


def preprocess_frame(frame, new_size=(84, 84)):
    if frame.ndim == 2:
        pass
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    else:
        raise ValueError("Unsupported frame shape: {}".format(frame.shape))
    # Anti_aliasing is used to reduce aliasing artifacts when resizing
    frame = resize(frame, new_size, anti_aliasing=True)
    return frame.astype(np.float32)

class FrameStacker:
    def __init__(self, env, k, preprocess_fn=preprocess_frame):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        self.preprocess_fn = preprocess_fn
        obs, _ = self.env.reset()
        processed_frame = self.preprocess_fn(obs)
        self.stacked_shape = (k, *processed_frame.shape)

        # Update observation space to reflect stacked frames
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.stacked_shape, dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self, seed=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        processed_obs = self.preprocess_fn(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return self._get_stacked_frames(), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        processed_next_obs = self.preprocess_fn(next_obs)
        self.frames.append(processed_next_obs)
        return self._get_stacked_frames(), reward, terminated, truncated, info

    def _get_stacked_frames(self):
        return np.array(self.frames)

    def render(self, *args, **kwargs):
        """Forward render calls to the underlying environment"""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Forward close calls to the underlying environment"""
        return self.env.close()

def ppo_update(policy, policy_optimizer, observations, actions, advantages, old_log_probs, config):
    clip_ratio = getattr(config, "clip_ratio", 0.2)
    policy_epochs = getattr(config, "policy_epochs", 4)
    minibatch_size = getattr(config, "minibatch_size", 256)
    entropy_coef = getattr(config, "value_loss_coef", 0.5)
    target_kl = getattr(config, "target_kl", 0.01)

    batch_size = observations.shape[0]

    for epoch in range(policy_epochs):
        # Create mini-batches
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            mb_indices = indices[start:end]

            mb_obs = observations[mb_indices]
            mb_actions = actions[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]

            dist = policy(mb_obs)

            if mb_actions.dim() == 1:
                # Discrete actions
                new_log_probs = dist.log_prob(mb_actions)
            else:
                # Continuous actions
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

            # new_probs / old_probs = exp(new_log_probs - mb_old_log_probs)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_advantages
            # Negative sign because we want to maximize the surrogate loss
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            # Negative entropy encourages exploration
            entropy_loss = -entropy_coef * entropy

            total_policy_loss = policy_loss + entropy_loss

            policy_optimizer.zero_grad()
            total_policy_loss.backward()

            # Optional
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

            policy_optimizer.step()

            # Early stopping based on KL divergence
            with torch.no_grad():
                # 
                kl_div = (mb_old_log_probs - new_log_probs).mean().item()
                if kl_div > target_kl:
                    print(f"Early stopping at epoch {epoch}, KL divergence: {kl_div}")
                    return
                
