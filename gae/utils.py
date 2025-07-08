import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import gymnasium as gym


def preprocess_frame(frame, new_size=(84, 84)):
    """Convert frame to grayscale and resize
    Assumes frame is NumPy already (H, W, C) or (H, W)"""
    if frame.ndim == 2:
        pass  # (H, W)
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)  # (H, W, 3) -> (H, W)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)  # (H, W, 1) -> (H, W)
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
