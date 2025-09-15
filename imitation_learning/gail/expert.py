import minari
import matplotlib.pyplot as plt
import time

# Load expert dataset
dataset = minari.load_dataset('mujoco/hopper/expert-v0', download=True)

# Recover environment in headless mode
env = dataset.recover_environment(render_mode="rgb_array")  # returns images instead of opening a window

# Get the first episode
episodes = list(dataset.iterate_episodes())
episode = episodes[0]
actions = episode.actions

# Store frames for visualization
frames = []

obs = env.reset()
done = False
step_idx = 0

while not done and step_idx < len(actions):
    action = actions[step_idx]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frame = env.render()  # returns an RGB image
    frames.append(frame)
    step_idx += 1

env.close()

import cv2
height, width, _ = frames[0].shape
out = cv2.VideoWriter('hopper_expert.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for f in frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
out.release()

