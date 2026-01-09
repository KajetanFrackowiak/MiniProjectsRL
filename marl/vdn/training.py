import wandb
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        env,
        agent,
        seed,
        optimizer,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        ckpt_manager,
        checkpoint_freq=10,
    ):
        self.env = env
        self.agent = agent
        self.seed = seed
        self.optimizer = optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.ckpt_manager = ckpt_manager
        self.checkpoint_freq = checkpoint_freq
        self.steps = 0

    def train_episode(self):
        reset_result = self.env.reset()

        # Handle different reset return types
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result

        if isinstance(obs, dict):
            # MPE - obs is already a dict from reset()
            agent_ids = sorted(obs.keys())
            observations = [np.array(obs[agent_id]) for agent_id in agent_ids]
            action_dict_func = lambda actions: {
                agent_id: actions[i] for i, agent_id in enumerate(agent_ids)
            }
            step_func = lambda action_dict: self.env.step(action_dict)
            reward_func = lambda rewards: sum(rewards.values())
            done_func = lambda dones, truncateds: any(dones.values()) or any(
                truncateds.values()
            )
        else:
            # SMAC - obs is a list from get_obs()
            agent_ids = list(range(len(obs)))
            observations = [np.array(o) for o in obs]
            action_dict_func = lambda actions: actions
            step_func = lambda actions: self.env.step(actions)
            reward_func = lambda reward: reward
            done_func = lambda terminated: terminated

        state = np.concatenate(observations)

        done = False
        episode_stats = {"reward": 0, "steps": 0, "loss": 0}
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        epsilon = max(self.epsilon_end, self.epsilon_start - self.steps * decay_rate)

        while not done:
            if isinstance(obs, dict):
                actions = self.agent.actions(observations, state, epsilon=epsilon)
            else:
                avail_actions = [self.env.get_avail_agent_actions(i) for i in agent_ids]
                actions = self.agent.actions(
                    observations, state, epsilon=epsilon, avail_actions=avail_actions
                )

            action_input = action_dict_func(actions)

            if isinstance(obs, dict):
                next_obs, rewards, dones, truncateds, info = step_func(action_input)
                next_observations = [
                    np.array(next_obs[agent_id]) for agent_id in agent_ids
                ]
                global_reward = reward_func(rewards)
                done = done_func(dones, truncateds)
            else:
                # SMAC
                reward, terminated, _ = step_func(action_input)
                next_obs = self.env.get_obs()
                next_observations = [np.array(o) for o in next_obs]
                global_reward = reward_func(reward)
                done = done_func(terminated)

            next_state = np.concatenate(next_observations)

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
            
            lr_attr = self.optimizer.learning_rate # EagerTensor
            current_lr = float(tf.convert_to_tensor(lr_attr))
            
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start
                - self.steps * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay),
            )

            tqdm.write(
                f"Ep {episode + 1}: Reward = {stats['reward']:.2f}, Steps = {stats['steps']}, Loss = {train_stats['loss'][-1]:.4f}, LR = {current_lr:.6f}, Epsilon = {epsilon:.6f}, Seed = {self.seed}"
            )

            wandb.log(
                {
                    "episode_reward": stats["reward"],
                    "episode_steps": stats["steps"],
                    "avg_loss": train_stats["loss"][-1],
                    "epsilon": epsilon,
                    "learning_rate": current_lr,
                },
                step=episode,
            )

            if (episode + 1) % self.checkpoint_freq == 0:
                self.ckpt_manager.save(episode + 1)

        return train_stats
