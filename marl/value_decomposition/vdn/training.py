import wandb
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        env,
        agent,
        optimizer,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        ckpt_manager,
        checkpoint_freq=10,
    ):
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.ckpt_manager = ckpt_manager
        self.checkpoint_freq = checkpoint_freq
        self.steps = 0

    def train_episode(self):
        obs_dict, info = self.env.reset()

        agent_ids = sorted(obs_dict.keys())
        observations = [np.array(obs_dict[agent_id]) for agent_id in agent_ids]

        state = np.concatenate(observations)

        done = False
        episode_stats = {"reward": 0, "steps": 0, "loss": 0}
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        epsilon = max(self.epsilon_end, self.epsilon_start - self.steps * decay_rate)

        while not done:
            actions = self.agent.actions(observations, state, epsilon=epsilon)

            action_dict = {agent_id: actions[i] for i, agent_id in enumerate(agent_ids)}

            obs_dict, reward_dict, done_dict, truncated_dict, info = self.env.step(
                action_dict
            )

            next_observations = [np.array(obs_dict[agent_id]) for agent_id in agent_ids]

            next_state = np.concatenate(next_observations)

            global_reward = sum(reward_dict.values())

            done = any(done_dict.values()) or any(truncated_dict.values())

            self.agent.store(
                observations,
                state,
                actions,
                global_reward,
                next_observations,
                next_state,
                done,
            )

            loss = self.agent.train_step()

            observations = next_observations
            state = next_state
            episode_stats["reward"] += global_reward
            episode_stats["steps"] += 1
            self.steps += 1
            episode_stats["loss"] += loss if loss is not None else 0

        return episode_stats

    def train(self, num_episodes):
        train_stats = {"reward": [], "steps": [], "loss": []}

        for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
            stats = self.train_episode()
            train_stats["reward"].append(stats["reward"])
            train_stats["steps"].append(stats["steps"])
            train_stats["loss"].append(
                stats["loss"] / stats["steps"] if stats["steps"] > 0 else 0
            )

            tqdm.write(
                f"Ep {episode + 1}: Reward = {stats['reward']:.2f}, Steps = {stats['steps']}, Loss = {train_stats['loss'][-1]:.4f}"
            )

            wandb.log(
                {
                    "episode_reward": stats["reward"],
                    "episode_steps": stats["steps"],
                    "avg_loss": train_stats["loss"][-1],
                    "epsilon": max(
                        self.epsilon_end,
                        self.epsilon_start
                        - self.steps
                        * (
                            (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
                        ),
                    ),
                },
                step=episode,
            )

            if (episode + 1) % self.checkpoint_freq == 0:
                self.ckpt_manager.save(episode + 1)

        return train_stats
