import argparse
from utils import plot_all_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory containing seed folders",
    )
    args = parser.parse_args()

    env_names = [
        "simple_spread_v3",
        "simple_speaker_listener_v4",
        "simple_adversary_v3",
    ]

    plot_all_envs(args.results_dir, env_names)


if __name__ == "__main__":
    main()
