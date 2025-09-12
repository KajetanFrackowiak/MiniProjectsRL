import gymnasium as gym
import argparse
from irl import apprenticeship_learning
from expert import collect_expert_trajectories
from policy import evaluate_learned_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=[1, 2],
        type=int,
        required=True,
        help="1: CartPole-v1, 2: MountainCar-v0",
    )
    args = parser.parse_args()
    envs = {1: "CartPole-v1", 2: "MountainCar-v0"}
    env_name = envs[args.env]
    print("Env_name: ", env_name)

    env = gym.make(env_name)
    expert_trajs = collect_expert_trajectories(
        env, num_trajectories=100
    )  # Reduced for testing

    print("\nStarting Apprenticeship Learning...")
    env = gym.make(env_name, render_mode="human")
    reward_weights = apprenticeship_learning(env, expert_trajs, iterations=20)
    print("Learned reward weights:", reward_weights)

    # Evaluate the learned policy using the same method as training
    print("\nEvaluating learned policy...")
    test_episodes = 10
    avg_reward = evaluate_learned_policy(
        env_name, reward_weights, test_episodes, render=True
    )
    print(f"Average reward over {test_episodes} evaluation episodes: {avg_reward:.2f}")
