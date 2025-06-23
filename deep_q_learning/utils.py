import random
import collections
import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.transform import resize

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

