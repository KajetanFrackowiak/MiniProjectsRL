import numpy as np
from features import feature_expectations, feature_vector
from policy import generate_policy
from scipy.optimize import minimize


def apprenticeship_learning(env, expert_trajs, gamma=0.99, iterations=20, epsilon=0.1):
    """Apprenticeship Learning as in Abbeel & Ng (2004)"""
    mu_E = feature_expectations(expert_trajs, gamma)
    mu_list = []

    print(f"Expert feature expectations shape: {mu_E.shape}")
    print(f"Expert feature expectations norm: {np.linalg.norm(mu_E):.6f}")

    # Calculate expert performance for comparison
    expert_rewards = [sum([r for _, _, r in traj]) for traj in expert_trajs]
    expert_avg_reward = np.mean(expert_rewards)
    print(f"Expert average reward: {expert_avg_reward:.2f}")

    # First policy: random policy (not zero reward)
    feat_dim = len(mu_E)
    reward_weights = np.random.normal(0, 0.1, feat_dim)
    reward_weights = reward_weights / np.linalg.norm(reward_weights)  # Normalize

    print("Generating initial policy...")
    learner_trajs = generate_policy(env, reward_weights)
    mu = feature_expectations(learner_trajs, gamma)
    mu_list.append(mu)

    print(f"Initial learner feature expectations norm: {np.linalg.norm(mu):.6f}")

    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration + 1}/{iterations} ===")

        # Solve the max-margin problem using quadratic programming
        # max t subject to: w^T(mu_E - mu_i) >= t for all i, ||w||_2 <= 1
        def objective(x):
            # x is [t, w1, w2, ..., wN], where w1, w2, ... are the reward weights
            t = x[0]
            return -t  # minimize -t to maximize t

        def constraint_margin(x, i):
            t, w = x[0], x[1:]
            # mu_e is the expert feature expectations
            # mu_list[i] is the feature expectations of the i-th learner policy
            return np.dot(w, mu_E - mu_list[i]) - t  # >= 0

        def constraint_norm(x):
            # we do it to ensure that our weights are within the unit ball
            w = x[1:]
            return 1 - np.dot(w, w)  # ||w||^2 <= 1

        # Set up constraints
        constraints = []
        for i in range(len(mu_list)):
            constraints.append(
                {"type": "ineq", "fun": lambda x, i=i: constraint_margin(x, i)}
            )
        constraints.append({"type": "ineq", "fun": constraint_norm})

        # Initial guess with better initialization
        x0 = np.zeros(feat_dim + 1)
        x0[0] = 0.01  # t
        if len(mu_list) > 0:
            # Initialize w based on current best direction
            direction = mu_E - mu_list[-1]
            if np.linalg.norm(direction) > 1e-8:
                x0[1:] = direction / np.linalg.norm(direction) * 0.5
            else:
                x0[1:] = np.random.normal(0, 0.1, feat_dim)
        else:
            x0[1:] = np.random.normal(0, 0.1, feat_dim)

        # Solve optimization with better options
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False, "maxiter": 200},
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")
            # Try with a different initial point
            x0[1:] = np.random.normal(0, 0.1, feat_dim)
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints=constraints,
                options={"ftol": 1e-8, "disp": False, "maxiter": 200},
            )
            if not result.success:
                print("Optimization failed again, continuing with current weights...")
                break

        t_opt = result.x[0]
        w_opt = result.x[1:]

        print(f"Margin: {t_opt:.6f}")
        print(f"Reward weights norm: {np.linalg.norm(w_opt):.6f}")
        print(
            f"Reward weights: {w_opt[:5]}..."
            if len(w_opt) > 5
            else f"Reward weights: {w_opt}"
        )

        # Check stopping criterion
        if t_opt <= epsilon:
            print(f"Converged! Margin {t_opt:.6f} <= epsilon {epsilon}")
            break

        # Generate new policy with learned reward
        print("Generating policy with learned reward...")
        learner_trajs = generate_policy(env, w_opt)
        mu = feature_expectations(learner_trajs, gamma)
        mu_list.append(mu)

        # Calculate and print average reward for this policy
        learner_rewards = [sum([r for _, _, r in traj]) for traj in learner_trajs]
        avg_reward = np.mean(learner_rewards)
        std_reward = np.std(learner_rewards)
        print(f"Learner average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Expert vs Learner gap: {expert_avg_reward - avg_reward:.2f}")

        # Check feature expectation difference
        feat_diff = np.linalg.norm(mu_E - mu)
        print(f"Feature expectation difference: {feat_diff:.6f}")

        reward_weights = w_opt

    print(f"\nFinal reward weights: {reward_weights}")
    return reward_weights
