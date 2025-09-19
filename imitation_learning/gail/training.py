import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os


class Trainer:
    def __init__(
        self,
        env,
        expert,
        policy_network,
        discriminator,
        policy_optimizer,
        disc_optimizer,
        policy_scheduler,
        disc_scheduler,
        epochs,
        episodes_per_epoch,
        checkpoint_interval,
        device,
        seed,
        project_name="GAIL",
        checkpoint_dir="checkpoints",
    ):
        self.env = env
        self.expert = expert
        self.policy_network = policy_network
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer
        self.policy_scheduler = policy_scheduler
        self.disc_scheduler = disc_scheduler
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

        self.training_metrics = {
            "epochs": [],
            "policy_losses": [],
            "disc_losses": [],
            "env_rewards_per_step": [],
            "gail_rewards": [],
            "total_episode_rewards": [],
            "episode_lengths": [],
            "normalized_performance": [],
        }

        # Calculate expert baseline for normalization
        self.expert_baseline = self._calculate_expert_baseline()
        self.random_baseline = 0.0  # Hopper random baseline

        wandb.init(project=project_name)

    def _calculate_expert_baseline(self):
        """Calculate expert performance baseline from demonstrations."""
        expert_rewards = []

        # Reset expert and run a few episodes to get baseline
        for _ in range(5):
            self.expert.reset()
            state, _ = self.env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 1000:  # Prevent infinite episodes
                expert_action = self.expert(
                    torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )
                action = expert_action.squeeze(0).cpu().numpy()
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                total_reward += reward
                state = next_state
                steps += 1

            expert_rewards.append(total_reward)

        baseline = (
            sum(expert_rewards) / len(expert_rewards) if expert_rewards else 1000.0
        )
        print(
            f"Expert baseline calculated: {baseline:.2f} (from {len(expert_rewards)} episodes)"
        )
        return baseline

    def train_step(self, state):
        self.policy_network.train()
        self.discriminator.train()

        with torch.no_grad():
            policy_output = self.policy_network(state)
            if self.policy_network.discrete:
                policy_action_disc = F.softmax(policy_output, dim=-1)
            else:
                mean, std = policy_output
                dist = torch.distributions.Normal(mean, std)
                policy_action_disc = dist.sample()

        expert_action = self.expert(state)

        D_policy_disc = self.discriminator(state, policy_action_disc)
        D_expert = self.discriminator(state, expert_action)

        disc_loss = -torch.mean(
            torch.log(D_expert + 1e-8) + torch.log(1 - D_policy_disc + 1e-8)
        )
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        policy_output = self.policy_network(state)
        if self.policy_network.discrete:
            policy_action_policy = F.softmax(policy_output, dim=-1)
        else:
            mean, std = policy_output
            dist = torch.distributions.Normal(mean, std)
            policy_action_policy = dist.rsample()

        D_policy_policy = self.discriminator(state, policy_action_policy)
        policy_loss = -torch.mean(torch.log(D_policy_policy + 1e-8))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        with torch.no_grad():
            gail_reward = torch.log(D_policy_policy + 1e-8)

        return {
            "gail_reward": gail_reward.mean().item(),
            "policy_loss": policy_loss.item(),
            "disc_loss": disc_loss.item(),
        }

    def train_one_episode(self):
        state, _ = self.env.reset()
        self.expert.reset()
        done = False
        episode_stats = {
            "env_reward_total": 0.0,
            "env_reward_per_step": 0.0,
            "gail_reward": 0.0,
            "policy_loss": 0.0,
            "disc_loss": 0.0,
            "steps": 0,
        }

        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            step_stats = self.train_step(state_tensor)

            with torch.no_grad():
                policy_output = self.policy_network(state_tensor)
                if self.policy_network.discrete:
                    probs = F.softmax(policy_output, dim=-1)
                    action_idx = torch.multinomial(probs, 1)
                    action = action_idx.squeeze().cpu().numpy()
                else:
                    mean, std = policy_output
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().squeeze(0).cpu().numpy()

            next_state, env_reward, done, truncated, _ = self.env.step(action)
            done = done or truncated

            episode_stats["env_reward_total"] += env_reward
            episode_stats["env_reward_per_step"] += env_reward
            episode_stats["gail_reward"] += step_stats["gail_reward"]
            episode_stats["policy_loss"] += step_stats["policy_loss"]
            episode_stats["disc_loss"] += step_stats["disc_loss"]
            episode_stats["steps"] += 1
            state = next_state
 
        for k in ["env_reward_per_step", "gail_reward", "policy_loss", "disc_loss"]:
            episode_stats[k] /= episode_stats["steps"]
        return episode_stats

    def train(self):
        pbar = tqdm(range(self.epochs), desc="Epochs")
        for epoch in pbar:
            all_episode_stats = []
            epoch_total_rewards = []
            epoch_lengths = []

            for _ in range(self.episodes_per_epoch):
                stats = self.train_one_episode()
                all_episode_stats.append(stats)
                epoch_total_rewards.append(stats["env_reward_total"])
                epoch_lengths.append(stats["steps"])

            # Calculate average stats for this epoch (per-step metrics)
            avg_stats = {
                k: sum(s[k] for s in all_episode_stats) / len(all_episode_stats)
                for k in all_episode_stats[0]
                if k not in ["steps", "env_reward_total"]
            }

            # Calculate episode-level metrics
            avg_total_reward = sum(epoch_total_rewards) / len(epoch_total_rewards)
            avg_episode_length = sum(epoch_lengths) / len(epoch_lengths)

            # Calculate normalized performance (paper metric)
            normalized_perf = (
                max(
                    0.0,
                    min(
                        1.0,
                        (avg_total_reward - self.random_baseline)
                        / (self.expert_baseline - self.random_baseline),
                    ),
                )
                if self.expert_baseline != self.random_baseline
                else 0.0
            )

            self.policy_scheduler.step()
            self.disc_scheduler.step()

            # Store paper-style metrics
            self.training_metrics["epochs"].append(epoch)
            self.training_metrics["policy_losses"].append(avg_stats["policy_loss"])
            self.training_metrics["disc_losses"].append(avg_stats["disc_loss"])
            self.training_metrics["env_rewards_per_step"].append(
                avg_stats["env_reward_per_step"]
            )
            self.training_metrics["gail_rewards"].append(avg_stats["gail_reward"])
            self.training_metrics["total_episode_rewards"].append(avg_total_reward)
            self.training_metrics["episode_lengths"].append(avg_episode_length)
            self.training_metrics["normalized_performance"].append(normalized_perf)

            # Enhanced logging
            log_data = {
                "epoch": epoch,
                "avg_episode_length": avg_episode_length,
                "total_episode_reward": avg_total_reward,
                "normalized_performance": normalized_perf,
                **avg_stats,
            }
            wandb.log(log_data)

            pbar.set_postfix(
                total_reward=f"{avg_total_reward:.1f}",
                normalized_perf=f"{normalized_perf:.3f}",
                env_reward_step=f"{avg_stats['env_reward_per_step']:.2f}",
                ep_len=f"{avg_episode_length:.0f}",
                policy_loss=f"{avg_stats['policy_loss']:.4f}",
                disc_loss=f"{avg_stats['disc_loss']:.4f}"
            )


            if epoch % self.checkpoint_interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "policy_network": self.policy_network.state_dict(),
                        "discriminator": self.discriminator.state_dict(),
                        "policy_optimizer": self.policy_optimizer.state_dict(),
                        "disc_optimizer": self.disc_optimizer.state_dict(),
                        "training_metrics": self.training_metrics,
                        "epoch": epoch,
                    },
                    f"checkpoints/checkpoint_epoch_{epoch}_seed_{self.seed}.pth",
                )

        return self.training_metrics
