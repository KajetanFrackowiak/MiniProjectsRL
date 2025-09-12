import random
import collections
import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import os
import re


def preprocess_frame(frame, new_size=(84, 84)):
    """
    Convert frame to grayscale and resize
    Assumes frame is NumPy already (H, W, C) or  (H, W)
    """
    if frame.ndim == 2:
        pass
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    else:
        raise ValueError("Frame must be 2 or 3 dimensional")

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

    def reset(self):
        obs, info = self.env.reset()
        processed_obs = self.preprocess_fn(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return self._get_stacked_frames(), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        processed_next_obs = self.preprocess_fn(next_obs)
        self.frames.append(processed_next_obs)
        return self._get_stacked_frames(), reward, done, info

    def _get_stacked_frames(self):
        return np.array(self.frames)


def find_latest_checkpoint(checkpoint_dir, model_base_name):
    latest_checkpoint_path = None
    latest_episode = -1
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    for f_name in os.listdir(checkpoint_dir):
        if f_name.startswith(model_base_name) and f_name.endswith(".pth"):
            match = re.search(r"_ep(\d+)\.pth$", f_name)
            if match:
                episode_num = int(match.group(1))
                if episode_num > latest_episode:
                    latest_episode = episode_num
                    latest_checkpoint_path = os.path.join(checkpoint_dir, f_name)

    if latest_checkpoint_path:
        return latest_checkpoint_path, latest_episode

    return None, 0


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.stack(state),
            torch.stack(
                action
            ).squeeze(),  # Squeeze to remove extra dim if action is single
            torch.stack(reward).squeeze(),
            torch.stack(next_state),
            torch.stack(done).squeeze(),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, mode="proportional"):
        assert mode in ["proportional", "rank_based"], (
            "Invalid mode for PrioritizedReplayBuffer"
        )
        self.capacity = capacity
        self.alpha = alpha
        self.mode = mode

        self.buffer = []
        self.pos = 0  # Current position in the buffer
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, priority=None):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        if priority is None:
            max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        else:
            max_priority = priority

        max_priority = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done, max_priority)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            # Replace the oldest data with new data
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.mode == "proportional":
            return self._sample_proportional(batch_size, beta)
        elif self.mode == "rank_based":
            return self._sample_rank_based(batch_size, beta)

    def _sample_proportional(self, batch_size, beta=0.4):
        prios = self.priorities[: len(self.buffer)]
        probs = prios**self.alpha
        probs /= probs.sum()

        # Where p is the probability of each sample
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return self._gather_samples(indices, weights)

    def _sample_rank_based(self, batch_size, beta=0.4):
        prios = self.priorities[: len(self.buffer)]
        ranks = prios.argsort()[
            ::-1
        ]  # Sort in descending order because higher priority means more likely to be sampled
        rank_weights = 1.0 / (np.arange(len(ranks)) + 1)
        probs = rank_weights / rank_weights.sum()

        sampled_ranks = np.random.choice(len(self.buffer), batch_size, p=probs)
        indices = ranks[sampled_ranks]
        weights = (len(self.buffer) * probs[sampled_ranks]) ** (-beta)
        weights /= weights.max()

        return self._gather_samples(indices, weights)

    def _gather_samples(self, indices, weights):
        # function to gather samples based on indices
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, max_priorities = zip(*batch)

        return (
            torch.stack(states),  # (batch_size, num_stacked_frames, height, width)
            torch.stack(actions).squeeze(),  # (batch_size, 1) -> (batch_size,)
            torch.stack(rewards).squeeze(),  # (batch_size, 1) -> (batch_size,)
            torch.stack(next_states),  # (batch_size, num_stacked_frames, height, width)
            torch.stack(dones).squeeze(),  # (batch_size, 1) -> (batch_size,)
            torch.tensor(indices, dtype=torch.int64),  # (batch_size,)
            torch.tensor(weights, dtype=torch.float32),  # (batch_size,)
        )

    def __len__(self):
        return len(self.buffer)

    def max_priority(self):
        if len(self.buffer) == 0:
            return 1.0
        return self.priorities.max()
    
    def update_priorities(self, indices, priorities):
        """
        Update the priorities of the samples at the given indices.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.buffer[idx] = (*self.buffer[idx][:-1], priority)
        # Ensure priorities are normalized
        self.priorities /= self.priorities.max() if self.priorities.max() > 0 else 1.0
        return self.priorities


class NStepTransitionBuffer:
    def __init__(self, base_buffer, n_step=3, gamma=0.99):
        self.base_buffer = base_buffer
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
    
        if len(self.n_step_buffer) < self.n_step:
            return
        
        R, next_s, d = 0, None, False
        for idx, (_, _, r, ns, dn) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            next_s = ns
            d = d or dn
            if d:
                break

        s, a, _, _, _ = self.n_step_buffer[0]
        max_priority = self.base_buffer.max_priority() if len(self.base_buffer) > 0 else 1.0
        self.base_buffer.push(s, a, R, next_s, d, max_priority)
    
    def sample(self, *args, **kwargs):
        return self.base_buffer.sample(*args, **kwargs)
    
    def update_priorities(self, *args, **kwargs):
        return self.base_buffer.update_priorities(*args, **kwargs)

    def __len__(self):
        return len(self.base_buffer)
