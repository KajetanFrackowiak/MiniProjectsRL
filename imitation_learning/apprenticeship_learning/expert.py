import gymnasium as gym
import numpy as np


def collect_expert_trajectories(env, num_trajectories=100):
    """Collect expert trajectories using improved heuristic policies"""
    all_rewards = []
    trajectories = []

    for trajectory in range(num_trajectories):
        trajectory_rewards = []
        obs, _ = env.reset()
        done = False
        traj = []
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes

        while not done and step_count < max_steps:
            # Choose expert action based on environment with improved heuristics
            if env.spec.id.startswith("CartPole"):
                # CartPole: More sophisticated policy considering velocity
                # obs[0] is position, obs[1] is velocity, obs[2] is angle, obs[3] is angular velocity
                angle = obs[2] 
                angular_velocity = obs[3]

                # Predictive control: consider where the pole will be
                # 0.02 = time step, adjust as needed
                predicted_angle = angle + 0.02 * angular_velocity
                action = 0 if predicted_angle < 0 else 1

            elif env.spec.id.startswith("MountainCar"):
                # MountainCar: Energy-based policy
                # obs[0] is position, obs[1] is velocity
                position = obs[0]
                velocity = obs[1]

                # If moving right and on right side, or if at right boundary, push right
                if (velocity > 0 and position > -0.5) or position > 0.3:
                    action = 2  # push right
                # If moving left and on left side, or if at left boundary, push left
                elif (velocity < 0 and position < -0.5) or position < -1.0:
                    action = 0  # push left
                # If near bottom and low velocity, build momentum in the direction of the slope
                elif abs(velocity) < 0.01 and position < 0:
                    action = 0  # push left to build momentum
                else:
                    # Coast or make small adjustments
                    action = 1  # no push
            else:
                raise NotImplementedError("Expert not defined for this environment")

            next_obs, reward, terminated, truncated, _ = env.step(action)
            trajectory_rewards.append(reward)
            traj.append((obs, action, reward))
            done = terminated or truncated
            step_count += 1

            if done:
                break
            obs = next_obs

        episode_reward = sum(trajectory_rewards)
        if trajectory % 20 == 0:
            print(
                f"Trajectory {trajectory + 1}: reward = {episode_reward:.2f}, steps = {step_count}"
            )

        trajectories.append(traj)
        all_rewards.append(episode_reward)

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(
        f"Expert performance: {avg_reward:.2f} ± {std_reward:.2f} over {num_trajectories} episodes"
    )
    return trajectories
