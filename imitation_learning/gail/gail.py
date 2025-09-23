import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=True):
        super().__init__()
        self.discrete = discrete

        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 100)

        if self.discrete:
            self.action_head = nn.Linear(100, action_dim)
        else:
            self.mean_head = nn.Linear(100, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        if self.discrete:
            logits = self.action_head(x)
            return logits
        else:
            mean = self.mean_head(x)
            std = torch.exp(self.log_std)
            # print(f"Mean: {mean}, Std: {std}")  # Debugging line
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("NaN detected in mean or std!")
                mean = mean.nan_to_num(0.0)
                std = std.nan_to_num(1.0)
            std = torch.clamp(
                std, 1e-6, 1.0
            )  # Prevent std from being too small or too large
            return mean, std


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)

        x = self.fc1(x)
        x = F.tanh(x)

        x = self.fc2(x)
        x = F.tanh(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        return x


def ppo_update(
    policy,
    policy_optimizer,
    observations,
    actions,
    advantages,
    old_log_probs,
    clip_ratio=0.2,
    policy_epochs=4,
    ppo_minibatch_size=256,
    entropy_coef=0.01,
    target_kl=0.01,
):
    # Detach inputs to prevent gradient flow through them
    observations = observations.detach()
    actions = actions.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()
    
    batch_size = observations.shape[0]
    total_policy_loss = 0.0
    num_updates = 0
    for epoch in range(policy_epochs):
        # Create mini-batches
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, ppo_minibatch_size):
            end = min(start + ppo_minibatch_size, batch_size)
            mb_indices = indices[start:end]

            mb_obs = observations[mb_indices]
            mb_actions = actions[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]

            # Calculate new policy distribution
            policy_output = policy(mb_obs)
            if hasattr(policy, "discrete") and policy.discrete:
                dist = torch.distributions.Categorical(logits=policy_output)
            else:
                mean, std = policy_output
                dist = torch.distributions.Normal(mean, std)

            # Handle different action dimensions properly
            if mb_actions.dim() == 1:
                new_log_probs = dist.log_prob(mb_actions)
            else:
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

            # Calculate ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # Calculate surrogate losses
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            entropy_loss = -entropy_coef * entropy

            # Total policy loss
            batch_total_loss = policy_loss + entropy_loss
            total_policy_loss += (
                policy_loss.item()
            )  # Track just the policy loss component
            num_updates += 1

            # Update policy
            policy_optimizer.zero_grad()
            batch_total_loss.backward()

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

            policy_optimizer.step()

            # Early stopping based on KL divergence
            with torch.no_grad():
                kl_div = (mb_old_log_probs - new_log_probs).mean()
                if kl_div > target_kl * 1.5:
                    # print(
                    #     f"Early stopping at epoch {epoch} due to high KL divergence: {kl_div:.4f}"
                    # )
                    # Return average loss so far
                    return total_policy_loss / num_updates if num_updates > 0 else 0.0

    # Return average policy loss across all updates
    return total_policy_loss / num_updates if num_updates > 0 else 0.0
