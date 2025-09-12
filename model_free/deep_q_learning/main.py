import torch
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import yaml
import secrets
import argparse
from tqdm import tqdm
from agent import (
    DQNAgent,
    DoubleDQNAgent,
    PrioritizedReplayAgent,
    PrioritizedDuelingAgent,
    RainbowAgent,
)
from utils import FrameStacker, find_latest_checkpoint


def load_hyperparameters():
    with open("hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="1: PongNoFrameskip-v42: BreakoutNoFrameskip-v43: SeaquestNoFrameskip-v4",
    )
    parser.add_argument("--seed", type=int, default=secrets.randbelow(2**32))
    parser.add_argument(
        "--method",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="1: DQN"
        "2: Double DQN"
        "3: Prioritized Replay DQN"
        "4: Prioritized Dueling DQN"
        "5: Rainbow",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default="none",
        help="Path to the model checkpoint to load, or 'none' to skip loading",
    )

    args = parser.parse_args()
    env_names = {
        1: "PongNoFrameskip-v4",
        2: "BreakoutNoFrameskip-v4",
        3: "SeaquestNoFrameskip-v4",
    }
    ENV_NAME = env_names[args.env] if args.env in env_names else "None"
    if args.method == 1:
        METHOD_NAME = "DQN"
    elif args.method == 2:
        METHOD_NAME = "Double DQN"
    elif args.method == 3:
        METHOD_NAME = "Prioritized Replay DQN"
    elif args.method == 4:
        METHOD_NAME = "Prioritized Dueling DQN"
    elif args.method == 5:
        METHOD_NAME = "Rainbow"

    config = load_hyperparameters()

    NUM_FRAMES_STACK = config["NUM_FRAMES_STACK"]
    INPUT_HEIGHT = config["INPUT_HEIGHT"]
    INPUT_WIDTH = config["INPUT_WIDTH"]
    LEARNING_RATE = config["LEARNING_RATE"]
    GAMMA = config["GAMMA"]
    EPSILON_START = config["EPSILON_START"]
    EPSILON_END = config["EPSILON_END"]
    EPSILON_DECAY_STEPS = config["EPSILON_DECAY_STEPS"]
    BUFFER_SIZE = config["BUFFER_SIZE"]
    BATCH_SIZE = config["BATCH_SIZE"]
    TARGET_UPDATE_FREQ_FRAMES = config["TARGET_UPDATE_FREQ_FRAMES"]
    LEARN_START_FRAME = config["LEARN_START_FRAME"]
    NUM_EPISODES = config["NUM_EPISODES"]
    MAX_STEPS_PER_EPISODE = config["MAX_STEPS_PER_EPISODE"]
    LOG_INTERVAL = config["LOG_INTERVAL"]
    SAVE_MODEL_INTERVAL = config["SAVE_MODEL_INTERVAL"]
    CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
    ALPHA = config["ALPHA"]
    BETA_START = config["BETA_START"]
    BETA_FRAMES = config["BETA_FRAMES"]
    MODE = config["MODE"]
    NOISY_NETS = config["NOISY_NETS"]
    VMIN = config["VMIN"]
    VMAX = config["VMAX"]
    N_ATOMS = config["N_ATOMS"]
    N_STEP = config["N_STEP"]
    N_STEP_GAMMA = config["N_STEP_GAMMA"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Initialization ---
    raw_env = gym.make(ENV_NAME)

    raw_env.reset(seed=args.seed)
    raw_env.observation_space.seed(args.seed)
    raw_env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = FrameStacker(raw_env, NUM_FRAMES_STACK)

    INPUT_DIMS_AGENT = (NUM_FRAMES_STACK, INPUT_HEIGHT, INPUT_WIDTH)
    NUM_ACTIONS = raw_env.action_space.n

    if METHOD_NAME == "DQN":
        agent = DQNAgent(
            input_dims=INPUT_DIMS_AGENT,
            num_actions=NUM_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_steps=EPSILON_DECAY_STEPS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ_FRAMES,
            noisy_nets=NOISY_NETS,
        )
    elif METHOD_NAME == "Double DQN":
        agent = DoubleDQNAgent(
            input_dims=INPUT_DIMS_AGENT,
            num_actions=NUM_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_steps=EPSILON_DECAY_STEPS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ_FRAMES,
            noisy_nets=NOISY_NETS,
        )
    elif METHOD_NAME == "Prioritized Replay DQN":
        agent = PrioritizedReplayAgent(
            input_dims=INPUT_DIMS_AGENT,
            num_actions=NUM_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_steps=EPSILON_DECAY_STEPS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ_FRAMES,
            alpha=ALPHA,
            beta_start=BETA_START,
            beta_frames=BETA_FRAMES,
            mode=MODE,
        )
    elif METHOD_NAME == "Prioritized Dueling DQN":
        agent = PrioritizedDuelingAgent(
            input_dims=INPUT_DIMS_AGENT,
            num_actions=NUM_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_steps=EPSILON_DECAY_STEPS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ_FRAMES,
            alpha=ALPHA,
            beta_start=BETA_START,
            beta_frames=BETA_FRAMES,
            mode=MODE,
            noisy_nets=NOISY_NETS,
        )
    elif METHOD_NAME == "Rainbow":
        agent = RainbowAgent(
            input_dims=INPUT_DIMS_AGENT,
            num_actions=NUM_ACTIONS,
            Vmin=VMIN,
            Vmax=VMAX,
            num_atoms=N_ATOMS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            target_update_freq=TARGET_UPDATE_FREQ_FRAMES,
            alpha=ALPHA,
            beta_start=BETA_START,
            beta_frames=BETA_FRAMES,
            device=None,
            n_step=N_STEP,
            n_step_gamma=N_STEP_GAMMA,
        )
    # --- Wandb Initialization ---
    wandb.init(
        project=f"{METHOD_NAME}_{ENV_NAME}_seed_{args.seed}",
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
            "seed": args.seed,
            "method": METHOD_NAME,
            "alpha": ALPHA,
            "beta_start": BETA_START,
            "beta_frames": BETA_FRAMES,
            "mode": MODE,
            "input_dims": INPUT_DIMS_AGENT,
            "num_actions": NUM_ACTIONS,
        },
    )

    print(f"Device: {agent.device}")
    print(f"Input Dims in {ENV_NAME}: {INPUT_DIMS_AGENT}")
    print(f"Number of Actions in {ENV_NAME}: {NUM_ACTIONS}")
    wandb.watch(agent.policy_net, log_freq=1000)

    episode_rewards = []
    total_frames_collected = 0

    starting_episode = 1
    if args.load_model != "none":
        latest_checkpoint_path, last_episode_completed = find_latest_checkpoint(
            CHECKPOINT_DIR, args.load_model
        )

    if args.load_model != "none" and latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        agent.load_model(latest_checkpoint_path)
        total_frames_collected = agent.steps_done
        starting_episode = last_episode_completed + 1
        print(f"Resuming training from episode {starting_episode}")
    else:
        print("No checkpoint found")

    # --- Training Loop ---
    # Wrap the episode loop with tqdm
    for episode in tqdm(
        range(starting_episode, NUM_EPISODES + 1), desc="Training Episodes"
    ):
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
                    wandb.log(
                        {
                            "loss": loss,
                            "total_frames": total_frames_collected,
                            "agent_steps": agent.steps_done,
                        }
                    )

            state = next_state
            current_episode_reward += reward
            episode_steps += 1

            if done:
                break

        total_agent_steps = agent.steps_done
        episode_rewards.append(current_episode_reward)
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        avg_loss_this_episode = (
            episode_loss_sum / episode_steps
            if episode_steps > 0 and total_frames_collected > LEARN_START_FRAME
            else 0
        )
        if not isinstance(agent, RainbowAgent):
            current_epsilon = agent.epsilon_end + (
                agent.epsilon_start - agent.epsilon_end
            ) * np.exp(-1.0 * total_agent_steps / agent.epsilon_decay_steps)
            current_epsilon = max(agent.epsilon_end, current_epsilon)
        else:
            current_epsilon = None

        log_dict = {
            "episode": episode,
            "episode_reward": current_episode_reward,
            "avg_reward_last_100": avg_reward_last_100,
            "epsilon": current_epsilon if not isinstance(agent, RainbowAgent) else None,
            "total_frames": total_frames_collected,
            "episode_steps": episode_steps,
            "agent_total_steps": total_agent_steps,
        }
        if total_frames_collected > LEARN_START_FRAME:
            log_dict["avg_episode_loss"] = avg_loss_this_episode

        wandb.log(log_dict)

        if episode % LOG_INTERVAL == 0:
            if current_epsilon:
                tqdm.write(
                    f"Episode: {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Total Frames: {total_frames_collected} | "
                    f"Reward: {current_episode_reward:.2f} | Avg Reward (100): {avg_reward_last_100:.2f} | "
                    f"Epsilon: {current_epsilon:.3f} | Avg Loss: {avg_loss_this_episode:.4f}"
                )
            else:
                tqdm.write(
                    f"Episode: {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Total Frames: {total_frames_collected} | "
                    f"Reward: {current_episode_reward:.2f} | Avg Reward (100): {avg_reward_last_100:.2f} | "
                    f"Avg Loss: {avg_loss_this_episode:.4f}"
                )

        if episode % SAVE_MODEL_INTERVAL == 0:
            checkpoint_name = (
                f"{METHOD_NAME}_{ENV_NAME}_seed_{args.seed}_ep_{episode}.pth"
            )
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            agent.save_model(checkpoint_path)
            tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    raw_env.close()
    print("Training finished.")

    # --- Plotting (local) ---
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("episode_rewards_pong.png")
    wandb.log({"episode_rewards_plot": wandb.Image(plt)})
    # plt.show()

    wandb.finish()


if __name__ == "__main__":
    main()
