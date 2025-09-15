import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


class Trainer:
    def __init__(self, env, expert, policy_network, discriminator,
                 policy_optimizer, disc_optimizer, policy_scheduler, disc_scheduler, epochs, checkpoint_interval, device, seed, project_name="GAIL"):
        self.env = env
        self.expert = expert
        self.policy_network = policy_network
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer
        self.policy_scheduler = policy_scheduler
        self.disc_scheduler = disc_scheduler
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.seed = seed
        self.policy_losses = []
        self.disc_losses = []

        wandb.init(project=project_name)
        

    def train_step(self, state):
        self.policy_network.train()
        self.discriminator.train()

        policy_action = self.policy_network(state)
        expert_action = self.expert(state)

        D_policy = self.discriminator(torch.cat([state, policy_action], dim=-1))
        D_expert = self.discriminator(torch.cat([state, expert_action], dim=-1))

        disc_loss = -torch.mean(torch.log(D_expert + 1e-8) + torch.log(1 - D_policy + 1e-8))
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        reward = -torch.log(1 - D_policy + 1e-8)
        policy_loss = -torch.mean(reward)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {"reward": reward.mean().item(), "policy_loss": policy_loss.item(), "disc_loss": disc_loss.item()}

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        episode_stats = {"reward": 0.0, "policy_loss": 0.0, "disc_loss": 0.0, "steps": 0}
        pbar = tqdm(desc="Episode", leave=False)

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            step_stats = self.train_step(state_tensor)

            with torch.no_grad():
                policy_action = self.policy_network(state_tensor)
                action = policy_action.squeeze(0).numpy()
            next_state, env_reward, done, truncated, _ = self.env.step(action)
            done = done or truncated

            episode_stats["reward"] += env_reward
            episode_stats["policy_loss"] += step_stats["policy_loss"]
            episode_stats["disc_loss"] += step_stats["disc_loss"]
            episode_stats["steps"] += 1
            state = next_state
            pbar.update(1)

        pbar.close()
        for k in ["reward", "policy_loss", "disc_loss"]:
            episode_stats[k] /= episode_stats["steps"]
        return episode_stats

    def train(self):
        for epoch in range(self.epochs):
            all_episode_stats = []
            for _ in range(self.episodes_per_epoch):
                stats = self.train_one_episode()
                all_episode_stats.append(stats)
            avg_stats = {k: sum(s[k] for s in all_episode_stats) / len(all_episode_stats)
                         for k in all_episode_stats[0]}

            self.policy_scheduler.step()
            self.disc_scheduler.step()

            wandb.log({"epoch": epoch, **avg_stats})
            print(f"Epoch {epoch}: reward {avg_stats['reward']:.2f}, "
                  f"policy_loss {avg_stats['policy_loss']:.4f}, disc_loss {avg_stats['disc_loss']:.4f}")
            
            self.policy_losses.append(avg_stats["policy_loss"])
            self.disc_losses.append(avg_stats["disc_loss"])


            if epoch % self.checkpoint_interval == 0:
                torch.save({
                    "policy_network": self.policy_network.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "policy_optimizer": self.policy_optimizer.state_dict(),
                    "disc_optimizer": self.disc_optimizer.state_dict(),
                }, f"checkpoints/checkpoint_epoch_{epoch}_seed_{self.seed}.pth")
