import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from episode_buffer import EpisodeReplayBuffer


class Trainer:
    """QMIX Trainer: Implements original paper's episode-based training."""

    def __init__(
        self,
        env,
        agents,
        mixer,
        mixer_optimizer,
        num_episodes=5000,
        max_steps_per_episode=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        batch_size=32,
        buffer_size=5000,
        train_interval=4,
        device="cuda",
        checkpoint_interval=100,
        checkpoint_dir="checkpoints",
        eval_interval=50,
        eval_episodes=5,
        eval_max_steps=None,
        target_update_freq=200,
        seed=42,
    ):
        self.env = env
        self.agents = agents
        self.mixer = mixer
        self.target_mixer = type(mixer)(
            n_agents=mixer.n_agents,
            state_dim=mixer.state_dim,
            embed_dim=mixer.embed_dim,
            hypernet_embed=mixer.hypernet_embed,
        ).to(mixer.hyper_w1[0].weight.device)
        self.target_mixer.load_state_dict(mixer.state_dict())
        self.target_mixer.eval()
        self.mixer_optimizer = mixer_optimizer
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode or 200
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_max_steps = eval_max_steps
        self.target_update_freq = target_update_freq
        self.seed = seed

        self.global_step = 0
        self.update_step = 0
        self.episodes_collected = 0

        self._set_seed(seed)

        # Initialize episode replay buffer
        is_smac = getattr(self.env, "is_smac", False)
        if is_smac:
            agent_names = list(self.agents.keys())
            obs_dims = [self.agents[agent_names[0]].obs_dim for _ in agent_names]
        else:
            obs_dims = {a: self.agents[a].obs_dim for a in self.agents.keys()}

        self.replay_buffer = EpisodeReplayBuffer(
            capacity=buffer_size,
            n_agents=len(self.agents),
            obs_dims=obs_dims,
            n_actions=list(self.agents.values())[0].act_dim,
            max_episode_len=self.max_steps_per_episode,
            state_dim=mixer.state_dim,  # Use mixer's state_dim
            device=device,
        )

        self.train_stats = {"ep_avg_rewards": [], "ep_avg_losses": [], "ep_steps": []}

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def compute_epsilon(self, step_idx):
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (step_idx / self.epsilon_decay_steps),
        )

    def collect_episode(self):
        """Collect a single episode and store in replay buffer."""
        is_smac = getattr(self.env, "is_smac", False)

        # Reset environment
        if is_smac:
            self.env.reset()
            obs_list = self.env.get_obs()
            agent_names = list(self.agents.keys())
            obs = {agent_names[i]: obs_list[i] for i in range(len(agent_names))}
            terminated = False
        else:
            episode_seed = self.seed + self.episodes_collected
            obs, info = self.env.reset(seed=episode_seed)
            done = {a: False for a in self.env.agents}
            agent_names = list(self.agents.keys())

        # Episode storage
        episode_obs = [[] for _ in range(len(agent_names))]
        episode_actions = [[] for _ in range(len(agent_names))]
        episode_rewards = []
        episode_dones = []
        episode_states = []  # Global state for SMAC
        total_reward = 0.0
        steps = 0
        eps = self.compute_epsilon(self.global_step)

        # Collect episode
        while True:
            if is_smac:
                if terminated:
                    break
                avail_actions = self.env.get_avail_actions()
                actions_list = []
                for i, agent_name in enumerate(agent_names):
                    action = self.agents[agent_name].act(obs[agent_name], eps)
                    if avail_actions[i][action] == 0:
                        available = [
                            a for a, avail in enumerate(avail_actions[i]) if avail == 1
                        ]
                        if available:
                            action = np.random.choice(available)
                    actions_list.append(action)

                reward, terminated, info = self.env.step(actions_list)
                next_obs_list = self.env.get_obs()
                next_obs = {
                    agent_names[i]: next_obs_list[i] for i in range(len(agent_names))
                }

                # Store step data
                for i, agent_name in enumerate(agent_names):
                    episode_obs[i].append(obs[agent_name])
                    episode_actions[i].append(actions_list[i])
                episode_rewards.append(reward)  # Global team reward
                episode_dones.append(float(terminated))
                episode_states.append(self.env.get_state())  # Store global state
                total_reward += reward
                obs = next_obs
            else:
                if all(done.values()):
                    break

                actions = {a: self.agents[a].act(obs[a], eps) for a in agent_names}
                next_obs, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )
                done = {a: terminations[a] or truncations[a] for a in agent_names}

                # Store step data - use normalized team reward
                for i, agent_name in enumerate(agent_names):
                    episode_obs[i].append(obs[agent_name])
                    episode_actions[i].append(actions[agent_name])

                team_reward = sum(rewards.values()) / len(
                    rewards
                )  # Normalize by n_agents
                episode_rewards.append(team_reward)
                episode_dones.append(float(all(done.values())))
                # For MPE, use concatenated observations as state
                concat_obs = np.concatenate([obs[a] for a in agent_names])
                episode_states.append(concat_obs)
                total_reward += team_reward
                obs = next_obs

            steps += 1
            self.global_step += 1
            if steps >= self.max_steps_per_episode:
                # Episode timed out - set last done to 0 (not truly terminal)
                if episode_dones:
                    episode_dones[-1] = 0.0
                break

        # Add episode to replay buffer
        self.replay_buffer.add_episode(
            episode_obs, episode_actions, episode_rewards, episode_dones, episode_states
        )
        self.episodes_collected += 1

        return total_reward, steps

    def _qmix_update(self):
        """Batched QMIX update on episode samples."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch of episodes
        (
            obs_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            states_batch,
            lengths_batch,
        ) = self.replay_buffer.sample(self.batch_size)

        agent_list = list(self.agents.values())
        n_agents = len(agent_list)
        device = self.device

        # Flatten batch for processing: [batch_size, max_len] -> [batch_size * max_len]
        batch_size, max_len = rewards_batch.shape
        flat_batch_size = batch_size * max_len

        # Use actual states from buffer: [batch_size, max_len, state_dim]
        state_flat = states_batch.view(flat_batch_size, -1)

        # Compute Q-values for each agent
        agent_qs_flat = []
        for i, agent in enumerate(agent_list):
            obs_flat = obs_batch[i].view(flat_batch_size, -1)  # [batch * len, obs_dim]
            act_flat = actions_batch[:, :, i].reshape(flat_batch_size)  # [batch * len]

            q_vals = agent.q_net(obs_flat)  # [batch * len, n_actions]
            q_chosen = q_vals.gather(1, act_flat.unsqueeze(1)).squeeze(
                1
            )  # [batch * len]
            agent_qs_flat.append(q_chosen)

        agent_qs_t = torch.stack(agent_qs_flat, dim=1)  # [batch * len, n_agents]

        # Mix Q-values
        q_tot_flat = self.mixer(agent_qs_t, state_flat).squeeze(1)  # [batch * len]
        q_tot = q_tot_flat.view(batch_size, max_len)  # [batch, len]

        # Compute target Q-values using NEXT observations
        # We need to shift observations by 1 timestep to get next_obs
        with torch.no_grad():
            agent_next_qs_flat = []
            for i, agent in enumerate(agent_list):
                # Shift observations: use obs[t+1] for computing target Q(s', a')
                # For the last step, use the same obs (will be masked anyway)
                obs_shifted = torch.cat(
                    [
                        obs_batch[i][:, 1:, :],  # [batch, len-1, obs_dim]
                        obs_batch[i][:, -1:, :],  # [batch, 1, obs_dim] - duplicate last
                    ],
                    dim=1,
                )  # [batch, len, obs_dim]

                obs_shifted_flat = obs_shifted.view(flat_batch_size, -1)
                next_q = agent.target_q_net(obs_shifted_flat).max(dim=1)[
                    0
                ]  # [batch * len]
                agent_next_qs_flat.append(next_q)

            agent_next_qs_t = torch.stack(
                agent_next_qs_flat, dim=1
            )  # [batch * len, n_agents]

            # Also shift states for target computation
            states_shifted = torch.cat(
                [states_batch[:, 1:, :], states_batch[:, -1:, :]], dim=1
            )
            state_shifted_flat = states_shifted.view(flat_batch_size, -1)

            q_tot_next_flat = self.target_mixer(
                agent_next_qs_t, state_shifted_flat
            ).squeeze(1)
            q_tot_next = q_tot_next_flat.view(batch_size, max_len)

            gamma = agent_list[0].gamma
            targets = rewards_batch + gamma * (1 - dones_batch) * q_tot_next

            # Clamp targets to prevent instability
            targets = torch.clamp(targets, -20.0, 20.0)

        # Mask out padding
        mask = torch.arange(max_len, device=device).unsqueeze(
            0
        ) < lengths_batch.unsqueeze(1)
        mask = mask.float()

        # Masked Huber loss (more stable than MSE)
        td_error = q_tot - targets
        huber_loss = torch.where(
            torch.abs(td_error) < 1.0, 0.5 * td_error**2, torch.abs(td_error) - 0.5
        )
        masked_loss = huber_loss * mask
        loss = masked_loss.sum() / mask.sum()

        # Optimize
        self.mixer_optimizer.zero_grad()
        for agent in agent_list:
            agent.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10.0)
        for agent in agent_list:
            torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), 10.0)

        self.mixer_optimizer.step()
        for agent in agent_list:
            agent.optimizer.step()
            if agent.scheduler:
                agent.scheduler.step()

        self.update_step += 1

        # Update target networks
        if self.update_step % self.target_update_freq == 0:
            for agent in agent_list:
                agent.update_target_network()
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return loss.item()

    def train(self):
        """Main training loop: collect episodes then update."""
        for episode in tqdm(range(1, self.num_episodes + 1)):
            # Collect episode
            avg_reward, steps = self.collect_episode()

            # Train every episode once buffer has enough data
            total_loss = 0.0
            loss_count = 0
            if len(self.replay_buffer) >= self.batch_size:
                # Single update per episode like original QMIX paper
                loss = self._qmix_update()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            eps = self.compute_epsilon(self.global_step)
            first_agent = list(self.agents.values())[0]
            current_lr = first_agent.get_lr()

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
                    f"Steps: {steps}, Epsilon: {eps:.2f}, LR: {current_lr:.6f}, Buffer: {len(self.replay_buffer)}"
                )

            # Checkpointing
            if episode % self.checkpoint_interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                for agent_name, agent in self.agents.items():
                    torch.save(
                        {
                            "model": agent.q_net.state_dict(),
                            "target_model": agent.target_q_net.state_dict(),
                            "optimizer": agent.optimizer.state_dict(),
                            "scheduler": (
                                agent.scheduler.state_dict()
                                if agent.scheduler
                                else None
                            ),
                            "episode": episode,
                            "epsilon": eps,
                            "seed": self.seed,
                        },
                        f"{self.checkpoint_dir}/{agent_name}_episode_{episode}_seed_{self.seed}.pth",
                    )

                torch.save(
                    {
                        "model": self.mixer.state_dict(),
                        "target_model": self.target_mixer.state_dict(),
                        "optimizer": self.mixer_optimizer.state_dict(),
                        "episode": episode,
                        "seed": self.seed,
                    },
                    f"{self.checkpoint_dir}/mixer_episode_{episode}_seed_{self.seed}.pth",
                )
                print(f"Saved checkpoint at episode {episode}")

            # Evaluation
            if self.eval_interval and episode % self.eval_interval == 0:
                eval_stats = self.evaluate(episode)
                wandb.log(eval_stats, step=episode)
                print(
                    f"Eval episode {episode}: Avg Reward {eval_stats['eval_avg_reward']:.2f}, "
                    f"Avg Steps {eval_stats['eval_avg_steps']:.1f}"
                )

            wandb.log(
                {
                    "ep_avg_reward": avg_reward,
                    "ep_avg_loss": avg_loss,
                    "ep_steps": steps,
                    "lr": current_lr,
                    "epsilon": eps,
                    "global_step": self.global_step,
                    "buffer_size": len(self.replay_buffer),
                },
                step=episode,
            )

            self.train_stats["ep_avg_rewards"].append(avg_reward)
            self.train_stats["ep_avg_losses"].append(avg_loss)
            self.train_stats["ep_steps"].append(steps)

        return self.train_stats

    @torch.no_grad()
    def evaluate(self, episode_idx):
        eval_rewards = []
        eval_steps = []
        is_smac = getattr(self.env, "is_smac", False)

        for eval_ep in range(self.eval_episodes):
            if is_smac:
                self.env.reset()
                obs_list = self.env.get_obs()
                agent_names = list(self.agents.keys())
                obs = {agent_names[i]: obs_list[i] for i in range(len(agent_names))}
                terminated = False
            else:
                eval_seed = (
                    self.seed + 100000 + episode_idx * self.eval_episodes + eval_ep
                )
                obs, info = self.env.reset(seed=eval_seed)
                done = {a: False for a in self.env.agents}

            total_rewards = {a: 0.0 for a in self.agents.keys()}
            steps = 0

            while True:
                if is_smac:
                    if terminated:
                        break
                    avail_actions = self.env.get_avail_actions()
                    agent_names = list(self.agents.keys())
                    actions_list = []
                    for i, agent_name in enumerate(agent_names):
                        action = self.agents[agent_name].act(obs[agent_name], eps=0.0)
                        # Mask unavailable actions
                        if avail_actions[i][action] == 0:
                            available = [
                                a
                                for a, avail in enumerate(avail_actions[i])
                                if avail == 1
                            ]
                            if available:
                                action = available[0]  # Greedy: pick first available
                        actions_list.append(action)

                    reward, terminated, info = self.env.step(actions_list)
                    next_obs_list = self.env.get_obs()
                    obs = {
                        agent_names[i]: next_obs_list[i]
                        for i in range(len(agent_names))
                    }

                    for agent_name in agent_names:
                        total_rewards[agent_name] += reward
                else:
                    if all(done.values()):
                        break
                    actions = {
                        a: self.agents[a].act(obs[a], eps=0.0) for a in self.env.agents
                    }
                    next_obs, rewards, terminations, truncations, infos = self.env.step(
                        actions
                    )
                    done = {
                        a: terminations[a] or truncations[a] for a in self.env.agents
                    }
                    for a in self.env.agents:
                        total_rewards[a] += rewards[a]
                    obs = next_obs

                steps += 1
                if self.eval_max_steps and steps >= self.eval_max_steps:
                    break

            avg_reward = sum(total_rewards.values()) / len(total_rewards)
            eval_rewards.append(avg_reward)
            eval_steps.append(steps)

        return {
            "eval_avg_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "eval_avg_steps": float(np.mean(eval_steps)) if eval_steps else 0.0,
        }
