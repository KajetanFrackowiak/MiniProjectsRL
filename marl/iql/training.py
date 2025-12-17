import wandb
import torch
import os
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        env,
        agents,
        num_episodes=1000,
        max_steps_per_episode=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        device="cuda",
        checkpoint_interval=100,
        checkpoint_dir="checkpoints",
        seed=42,
    ):
        self.env = env
        self.agents = agents
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        self.train_stats = {"ep_avg_rewards": [], "ep_avg_losses": [], "ep_steps": []}

    def compute_epsilon(self, episode_idx):
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (episode_idx / self.epsilon_decay),
        )

    def train_episode(self, episode_idx):
        episode_stats = {"ep_avg_reward": 0, "ep_avg_loss": 0, "ep_steps": 0}
        episode_seed = self.seed + episode_idx
        obs, info = self.env.reset(seed=episode_seed)

        done = {a: False for a in self.env.agents}
        total_rewards = {a: 0.0 for a in self.env.agents}
        steps = 0
        eps = self.compute_epsilon(episode_idx)
        total_loss = 0.0
        loss_count = 0

        # all(done.values) checks if all agents are done
        while not all(done.values()):
            actions = {a: self.agents[a].act(obs[a], eps) for a in self.env.agents}

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            done = {a: terminations[a] or truncations[a] for a in self.env.agents}

            for a in self.env.agents:
                # Clip rewards to the range [-1, 1] for stability
                clipped_reward = max(min(rewards[a], 1.0), -1.0)
                # Each agent stores its own experience in its replay buffer
                self.agents[a].store(
                    obs[a], actions[a], clipped_reward, next_obs[a], done[a]
                )
                # Per-agent Q-learning update (inside train_step method)
                loss = self.agents[a].train_step()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                total_rewards[a] += rewards[a]

            obs = next_obs
            steps += 1
            if self.max_steps_per_episode and steps >= self.max_steps_per_episode:
                break

        avg_reward = sum(total_rewards.values()) / len(total_rewards)
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        episode_stats["ep_avg_reward"] = avg_reward
        episode_stats["ep_avg_loss"] = avg_loss
        episode_stats["ep_steps"] = steps

        return episode_stats

    def train(self):
        for episode in tqdm(range(1, self.num_episodes + 1)):
            episode_stats = self.train_episode(episode)

            first_agent = list(self.agents.values())[0]
            current_lr = first_agent.get_lr()
            eps = self.compute_epsilon(episode)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Average Reward: {episode_stats['ep_avg_reward']:.2f}, "
                    f"Average Loss: {episode_stats['ep_avg_loss']:.4f}, Steps: {episode_stats['ep_steps']}, "
                    f"Epsilon: {eps:.2f}, LR: {current_lr:.6f}, Seed: {self.seed}"
                )

            if episode % self.checkpoint_interval == 0:
                for agent_name, agent in self.agents.items():
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    checkpoint_path = f"{self.checkpoint_dir}/{agent_name}_episode_{episode}_seed_{self.seed}.pth"
                    torch.save(
                        {
                            "model": agent.q_net.state_dict(),
                            "optimizer": agent.optimizer.state_dict(),
                            "scheduler": agent.scheduler.state_dict()
                            if agent.scheduler
                            else None,
                            "episode": episode,
                            "epsilon": eps,
                            "seed": self.seed,
                        },
                        checkpoint_path,
                    )
                    print(
                        f"Saved checkpoint for {agent_name} at episode {episode} to {checkpoint_path}"
                    )

            wandb.log(episode_stats)
            wandb.log({"lr": current_lr, "epsilon": eps}, step=episode)

            self.train_stats["ep_avg_rewards"].append(episode_stats["ep_avg_reward"])
            self.train_stats["ep_avg_losses"].append(episode_stats["ep_avg_loss"])
            self.train_stats["ep_steps"].append(episode_stats["ep_steps"])

        return self.train_stats
