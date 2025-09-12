import numpy as np
from features import feature_vector


def discretize_state(obs, env_name, bins=15):
    """Discretize continuous state space"""
    if env_name.startswith("CartPole"):
        # CartPole bounds: pos[-4.8,4.8], vel[-inf,inf], angle[-0.418,0.418], ang_vel[-inf,inf]
        bounds = [(-4.8, 4.8), (-5, 5), (-0.418, 0.418), (-5, 5)]
    elif env_name.startswith("MountainCar"):
        # MountainCar bounds: pos[-1.2,0.6], vel[-0.07,0.07]
        bounds = [(-1.2, 0.6), (-0.07, 0.07)]
    else:
        raise NotImplementedError("Environment not supported")

    state_idx = []
    for i, (obs_i, (low, high)) in enumerate(zip(obs, bounds)):
        # Clip to bounds and discretize
        obs_i = np.clip(obs_i, low, high)
        idx = int((obs_i - low) / (high - low) * (bins - 1))
        idx = np.clip(idx, 0, bins - 1)
        state_idx.append(idx)

    # Convert to single state index
    state = 0
    for i, idx in enumerate(state_idx):
        state += idx * (bins**i)
    return state


def state_to_obs(state_idx, env_name, bins=15):
    """Convert discrete state index back to continuous observation"""
    if env_name.startswith("CartPole"):
        bounds = [(-4.8, 4.8), (-5, 5), (-0.418, 0.418), (-5, 5)]
        n_dims = 4
    elif env_name.startswith("MountainCar"):
        bounds = [(-1.2, 0.6), (-0.07, 0.07)]
        n_dims = 2
    else:
        raise NotImplementedError("Environment not supported")

    # Convert single index to multi-dimensional indices
    indices = []
    for i in range(n_dims):
        indices.append(state_idx % bins)
        state_idx //= bins

    # Convert indices to continuous values (center of bin)
    obs = []
    for i, (idx, (low, high)) in enumerate(zip(indices, bounds)):
        val = low + (idx + 0.5) * (high - low) / bins
        obs.append(val)

    return np.array(obs)


def build_transition_model(env, env_name, bins=20, n_samples=1000):
    """Build transition model P[s][a] = [(prob, next_state, reward, done)]"""
    n_actions = env.action_space.n
    n_states = bins ** (4 if env_name.startswith("CartPole") else 2)

    # Initialize transition model
    P = {}
    for s in range(n_states):
        P[s] = {}
        for a in range(n_actions):
            P[s][a] = []

    # Sample transitions for each state-action pair
    print("Building transition model...")
    for s in range(n_states):
        if s % 1000 == 0:
            print(f"State {s}/{n_states}")

        obs = state_to_obs(s, env_name, bins)

        for a in range(n_actions):
            next_states = {}

            for _ in range(n_samples):
                # Reset environment to current state (approximate)
                env.reset()
                env.unwrapped.state = obs if env_name.startswith("CartPole") else obs

                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                # Discretize next state
                if done:
                    next_s = -1  # Terminal state
                else:
                    next_s = discretize_state(next_obs, env_name, bins)

                # Count transitions
                if next_s not in next_states:
                    next_states[next_s] = {"count": 0, "reward": 0}
                next_states[next_s]["count"] += 1
                next_states[next_s]["reward"] += reward

            # Convert counts to probabilities
            total_count = sum(data["count"] for data in next_states.values())
            for next_s, data in next_states.items():
                prob = data["count"] / total_count
                avg_reward = data["reward"] / data["count"]
                done = next_s == -1
                P[s][a].append((prob, next_s, avg_reward, done))

    return P


def value_iteration(
    env_name, reward_weights, bins=15, gamma=0.99, theta=1e-4, max_iterations=100
):
    """Value iteration with learned reward function"""
    n_actions = 3 if env_name.startswith("MountainCar") else 2
    n_states = bins ** (4 if env_name.startswith("CartPole") else 2)

    # Initialize value function
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()

        for s in range(n_states):
            obs = state_to_obs(s, env_name, bins)
            features = feature_vector(obs)
            immediate_reward = np.dot(reward_weights, features)

            # Compute Q-values for each action by sampling transitions
            q_values = []
            for a in range(n_actions):
                # Sample multiple transitions to estimate expected future value
                expected_value = 0
                n_samples = 5  # Reduced for efficiency

                for _ in range(n_samples):
                    # Simulate transition (simplified)
                    if env_name.startswith("CartPole"):
                        next_obs = simulate_cartpole_transition(obs, a)
                        # Check for terminal conditions
                        if abs(next_obs[0]) > 2.4 or abs(next_obs[2]) > 0.2095:
                            next_value = 0  # Terminal state
                        else:
                            next_s = discretize_state(next_obs, env_name, bins)
                            next_value = V_old[next_s]
                    elif env_name.startswith("MountainCar"):
                        next_obs = simulate_mountaincar_transition(obs, a)
                        # Check for terminal conditions
                        if next_obs[0] >= 0.5:  # Reached goal
                            next_value = 0  # Terminal state
                        else:
                            next_s = discretize_state(next_obs, env_name, bins)
                            next_value = V_old[next_s]

                    expected_value += next_value

                expected_value /= n_samples
                q_values.append(immediate_reward + gamma * expected_value)

            V[s] = max(q_values)
            delta = max(delta, abs(V_old[s] - V[s]))

        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations")
            break
        print(f"Iteration: {iteration}")

    # Extract optimal policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        obs = state_to_obs(s, env_name, bins)
        features = feature_vector(obs)
        immediate_reward = np.dot(reward_weights, features)

        q_values = []
        for a in range(n_actions):
            # Sample transitions for policy extraction
            expected_value = 0
            n_samples = 5

            for _ in range(n_samples):
                if env_name.startswith("CartPole"):
                    next_obs = simulate_cartpole_transition(obs, a)
                    if abs(next_obs[0]) > 2.4 or abs(next_obs[2]) > 0.2095:
                        next_value = 0
                    else:
                        next_s = discretize_state(next_obs, env_name, bins)
                        next_value = V[next_s]
                elif env_name.startswith("MountainCar"):
                    next_obs = simulate_mountaincar_transition(obs, a)
                    if next_obs[0] >= 0.5:
                        next_value = 0
                    else:
                        next_s = discretize_state(next_obs, env_name, bins)
                        next_value = V[next_s]

                expected_value += next_value

            expected_value /= n_samples
            q_values.append(immediate_reward + gamma * expected_value)

        policy[s] = np.argmax(q_values)

    return policy


def simulate_cartpole_transition(obs, action):
    """Simplified CartPole dynamics simulation"""
    # CartPole physics approximation
    x, x_dot, theta, theta_dot = obs
    force = 10.0 if action == 1 else -10.0

    # Simplified dynamics (approximate)
    dt = 0.02
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Approximate dynamics
    temp = (force + 0.1 * theta_dot * theta_dot * sin_theta) / 1.1
    theta_acc = (9.8 * sin_theta - cos_theta * temp) / (
        0.5 * (4.0 / 3.0 - 0.1 * cos_theta * cos_theta / 1.1)
    )
    x_acc = temp - 0.1 * theta_acc * cos_theta / 1.1

    # Update state
    x = x + dt * x_dot
    x_dot = x_dot + dt * x_acc
    theta = theta + dt * theta_dot
    theta_dot = theta_dot + dt * theta_acc

    return np.array([x, x_dot, theta, theta_dot])


def simulate_mountaincar_transition(obs, action):
    """Simplified MountainCar dynamics simulation"""
    position, velocity = obs

    # MountainCar dynamics
    force = 0.001 * (action - 1)  # -1, 0, or 1
    velocity += force - 0.0025 * np.cos(3 * position)
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)

    # Reset velocity if hit left bound
    if position == -1.2 and velocity < 0:
        velocity = 0

    return np.array([position, velocity])


def generate_policy(env, reward_weights, episodes=10, gamma=0.99):
    """Generate policy using value iteration (as in the paper)"""
    env_name = env.spec.id
    bins = 15  # Use same bin count as evaluation

    # Get optimal policy using value iteration
    policy = value_iteration(env_name, reward_weights, bins, gamma)
    print(f'Policy: {policy}')
    # Generate trajectories using the optimal policy
    trajectories = []
    for _ in range(episodes):
        obs, _ = env.reset()
        traj = []
        done = False
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes

        while not done and step_count < max_steps:
            # Discretize current state and get action from policy
            state_idx = discretize_state(obs, env_name, bins)
            action = policy[state_idx]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            traj.append((obs, action, reward))
            done = terminated or truncated
            step_count += 1

            if done:
                break
            obs = next_obs

        trajectories.append(traj)

    return trajectories


def evaluate_learned_policy(
    env_name, reward_weights, episodes=10, render=False, bins=15
):
    """Evaluate the learned policy using value iteration"""
    import gymnasium as gym

    # Create environment
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Get optimal policy
    policy = value_iteration(env_name, reward_weights, bins)

    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes

        while not done and step_count < max_steps:
            # Get action from learned policy
            state_idx = discretize_state(obs, env_name, bins)
            action = policy[state_idx]

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1

            if render and ep == 0:  # Only render first episode
                env.render()

        total_rewards.append(episode_reward)
        if ep % 5 == 0:
            print(f"Episode {ep + 1}: reward = {episode_reward:.2f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Policy evaluation: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward
