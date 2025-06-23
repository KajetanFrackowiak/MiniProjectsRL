# train.py
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent
from utils import preprocess_frame
from collections import deque
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
import re

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
            match = re.search(r'_ep(\d+)\.pth$', f_name)
            if match:
                episode_num = int(match.group(1))
                if episode_num > latest_episode:
                    latest_episode = episode_num
                    latest_checkpoint_path = os.path.join(checkpoint_dir, f_name)

    if latest_checkpoint_path:
        return latest_checkpoint_path, latest_episode

    return None, 0


def main():
    # --- Hyperparameters ---
    ENV_NAME = "PongNoFrameskip-v4"
    NUM_FRAMES_STACK = 4
    INPUT_HEIGHT = 84
    INPUT_WIDTH = 84

    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_STEPS = 1000000
    BUFFER_SIZE = 20000  # Reduced for faster testing, original was 100_000
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ_FRAMES = 10000
    LEARN_START_FRAME = 50000

    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 10000
    LOG_INTERVAL = 10
    SAVE_MODEL_INTERVAL = 100
    CHECKPOINT_DIR = "checkpoints"
    MODEL_BASE_NAME = f"{ENV_NAME}_dqn"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Wandb Initialization ---
    wandb.init(
        project="dqn-pong",
        config={
            "environment": ENV_NAME,
            "num_frames_stack": NUM_FRAMES_STACK,
            "input_height": INPUT_HEIGHT,
            "input_width": INPUT_WIDTH,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay_steps": EPSILON_DECAY_STEPS,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "target_update_freq_frames": TARGET_UPDATE_FREQ_FRAMES,
            "learn_start_frame": LEARN_START_FRAME,
            "num_episodes": NUM_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        }
    )

    # --- Initialization ---
    raw_env = gym.make(ENV_NAME, render_mode=None)
    env = FrameStacker(raw_env, NUM_FRAMES_STACK, preprocess_fn=preprocess_frame)

    input_dims_agent = (NUM_FRAMES_STACK, INPUT_HEIGHT, INPUT_WIDTH)
    num_actions = raw_env.action_space.n

    agent = DQNAgent(input_dims=input_dims_agent,
                     num_actions=num_actions,
                     learning_rate=LEARNING_RATE,
                     gamma=GAMMA,
                     epsilon_start=EPSILON_START,
                     epsilon_end=EPSILON_END,
                     epsilon_decay_steps=EPSILON_DECAY_STEPS,
                     buffer_size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE,
                     target_update_freq=TARGET_UPDATE_FREQ_FRAMES
                     )

    print(f"Device: {agent.device}")
    print(f"Input Dims for DQN: {input_dims_agent}")
    print(f"Number of Actions: {num_actions}")
    wandb.watch(agent.policy_net, log_freq=1000)

    episode_rewards = []
    total_frames_collected = 0

    starting_episode = 1
    latest_checkpoint_path, last_episode_completed = find_latest_checkpoint(CHECKPOINT_DIR, MODEL_BASE_NAME)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        agent.load_model(latest_checkpoint_path)
        total_frames_collected = agent.steps_done
        starting_episode = last_episode_completed + 1
        print(f"Resuming training from episode {starting_episode}")
    else:
        print("No checkpoint found")

    # --- Training Loop ---
    # Wrap the episode loop with tqdm
    for episode in tqdm(range(starting_episode, NUM_EPISODES + 1), desc="Training Episodes"):
        state, _ = env.reset()
        current_episode_reward = 0
        episode_loss_sum = 0
        episode_steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            total_frames_collected += 1

            loss = None
            if total_frames_collected > LEARN_START_FRAME:
                loss = agent.learn()
                if loss is not None:
                    episode_loss_sum += loss
                    wandb.log({"loss": loss, "total_frames": total_frames_collected, "agent_steps": agent.steps_done})

            state = next_state
            current_episode_reward += reward
            episode_steps += 1

            if done:
                break

        total_agent_steps = agent.steps_done
        episode_rewards.append(current_episode_reward)
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        avg_loss_this_episode = episode_loss_sum / episode_steps if episode_steps > 0 and total_frames_collected > LEARN_START_FRAME else 0
        current_epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                          np.exp(-1. * total_agent_steps / agent.epsilon_decay_steps)
        current_epsilon = max(agent.epsilon_end, current_epsilon)

        log_dict = {
            "episode": episode,
            "episode_reward": current_episode_reward,
            "avg_reward_last_100": avg_reward_last_100,
            "epsilon": current_epsilon,
            "total_frames": total_frames_collected,
            "episode_steps": episode_steps,
            "agent_total_steps": total_agent_steps,
        }
        if total_frames_collected > LEARN_START_FRAME:
            log_dict["avg_episode_loss"] = avg_loss_this_episode

        wandb.log(log_dict)

        if episode % LOG_INTERVAL == 0:
            tqdm.write(f"Episode: {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Total Frames: {total_frames_collected} | "
                       f"Reward: {current_episode_reward:.2f} | Avg Reward (100): {avg_reward_last_100:.2f} | "
                       f"Epsilon: {current_epsilon:.3f} | Avg Loss: {avg_loss_this_episode:.4f}")

        if episode % SAVE_MODEL_INTERVAL == 0:
            checkpoint_name = f"{MODEL_BASE_NAME}_ep{episode}.pth"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            agent.save_model(checkpoint_path)
            tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    raw_env.close()
    print("Training finished.")

    # --- Plotting (local) ---
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('episode_rewards_pong.png')
    wandb.log({"episode_rewards_plot": wandb.Image(plt)})
    # plt.show()

    wandb.finish()


if __name__ == "__main__":
    main()