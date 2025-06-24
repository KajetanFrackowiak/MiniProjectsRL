import torch
import torch.nn.functional as F
import numpy as np

# Utility to flatten model parameters to a single vector
def flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

# Utility to set model parameters from a flat vector
def set_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[idx:idx+n].view(p.size()))
        idx += n

# Compute flattened gradients of a scalar loss w.r.t model parameters
def flat_grad(loss, model, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([grad.contiguous().view(-1) for grad in grads])

# Conjugate gradient solver for linear system Ax = b (A implicit)
def conjugate_gradient(Avp_func, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

# Backtracking line search to satisfy improvement and KL constraints
def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = f().item()
    for stepfrac in 0.5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        set_params(model, xnew)
        newfval = f().item()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x

# Surrogate loss (policy objective) for TRPO
def surrogate_loss(policy, observations, actions, advantages, old_log_probs):
    dist = policy(observations)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(new_log_probs - old_log_probs)
    return -(ratio * advantages).mean()

# Mean KL divergence between old and new policy distributions
def mean_kl_divergence(policy, observations, old_dist):
    new_dist = policy(observations)
    kl = torch.distributions.kl_divergence(old_dist, new_dist)
    return kl.mean()

# TRPO policy update step with KL constraint
def trpo_update(policy, observations, actions, advantages, old_log_probs, max_kl=1e-2, damping=1e-2):
    # Calculate surrogate loss and gradients
    loss = surrogate_loss(policy, observations, actions, advantages, old_log_probs)
    grads = flat_grad(loss, policy, retain_graph=True)

    # Fisher-vector product function for conjugate gradient
    def Fvp(v):
        with torch.no_grad():
            old_dist = policy(observations)
        kl = mean_kl_divergence(policy, observations, old_dist)
        grads_kl = flat_grad(kl, policy, create_graph=True, retain_graph=True)
        return flat_grad((grads_kl * v).sum(), policy, retain_graph=True) + damping * v

    stepdir = conjugate_gradient(Fvp, -grads)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]
    expected_improve = -(grads * stepdir).sum(0, keepdim=True) / lm[0]

    old_params = flat_params(policy)
    success, new_params = linesearch(policy, lambda: surrogate_loss(policy, observations, actions, advantages, old_log_probs), old_params, fullstep, expected_improve)
    if not success:
        set_params(policy, old_params)
        print("Line search failed. Using old parameters.")
    else:
        set_params(policy, new_params)

def ppo_update(policy, value_fn, optimizer_policy, observations,
               actions, advantages, returns, old_log_probs, clip_epsilon=0.2, c1=0.5, c2=0.01, epochs=10, batch_size=64):
    dataset_size = observations.size(0)  # N * T from paper

    for _ in range(epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            obs_batch = observations[batch_idx]
            act_batch = actions[batch_idx]
            adv_batch = advantages[batch_idx]
            ret_batch = returns[batch_idx]
            old_logp_batch = old_log_probs[batch_idx]

            # Policy forward pass
            dist = policy(obs_batch)
            new_log_probs = dist.log_prob(act_batch).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Clipped objective
            ratio = torch.exp(new_log_probs - old_logp_batch)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * adv_batch, clipped_ratio * adv_batch).mean()

            # Value function loss
            value_preds = value_fn(obs_batch).squeeze(-1)
            value_loss = F.mse_loss(value_preds, ret_batch)

            # Total loss with entropy bonus
            loss = policy_loss - c1 * value_loss + c2 * entropy.mean()

            optimizer_policy.zero_grad()
            loss.backward()
            optimizer_policy.step()


# Compute Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0]  # append 0 for terminal state value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

# Update value function with MSE loss
def update_value_function(value_fn, optimizer, observations, returns, epochs=5):
    for _ in range(epochs):
        values_pred = value_fn(observations)
        loss = F.mse_loss(values_pred, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
