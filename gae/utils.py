import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess_frame(frame, new_size=(84,84)):
    """Convert frame to grayscale and resize
    Assumes frame is NumPy already (H, W, C) or (H, W)"""
    if frame.ndim == 2:
        pass  # (H, W)
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)  # (H, W, 3) -> (H, W)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)   # (H, W, 1) -> (H, W)
    else:
        raise ValueError('Frame must be 2 or 3 dimensional')

    frame = resize(frame, new_size, anti_aliasing=True)
    return frame.astype(np.float32)