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
        raise ValueError('Frame must be 2 or 3 dimensional')

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
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(state),
                torch.stack(action).squeeze(),  # Squeeze to remove extra dim if action is single
                torch.stack(reward).squeeze(),
                torch.stack(next_state),
                torch.stack(done).squeeze()
        )

    def __len__(self):
        return len(self.buffer)


