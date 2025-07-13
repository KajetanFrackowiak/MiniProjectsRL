import numpy as np
import random
import torch
import cv2
from collections import deque


def preprocess_frame(frame, new_size=(84, 84)):
    if frame.ndim == 2:
        pass
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    else:
        raise ValueError("Frame must be 2 or 3 dimensional")
    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32)


class FrameStacker:
    def __init__(self, env, k, preprocess_frame=preprocess_frame):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        self.preprocess_frame = preprocess_frame
        obs, _ = self.env.reset()
        self.processed_frame = self.preprocess_frame(obs)  # (84, 84)
        self.stacked_shape = (k, *self.processed_frame.shape)

    def reset(self):
        obs, info = self.env.reset()
        processed_frame = self.preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(processed_frame)
        # frames.shape (k, 84, 84)
        return self._get_stacked_frames(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_frame = self.preprocess_frame(obs)
        self.frames.append(processed_frame)
        return self._get_stacked_frames(), reward, terminated, truncated, info

    def _get_stacked_frames(self):
        return np.array(self.frames)

    def get_obs_dim(self):
        return self.processed_frame.shape
    
    def get_act_dim(self):
        return self.env.action_space.n


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.rng = np.random.default_rng()

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in state]),  # (batch_size, k, 84, 84)
            torch.stack([torch.tensor([a], dtype=torch.int64) for a in action]).squeeze(),  # (batch_size,)
            torch.stack([torch.tensor([r], dtype=torch.float32) for r  in reward]).squeeze(),  # (batch_size,)
            torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_state]),  # (batch_size, k, 84, 84)
            torch.stack([torch.tensor([d], dtype=torch.float32) for d in done]).squeeze(),  # (batch_size,)
        )

    def __len__(self):
        return len(self.buffer)
