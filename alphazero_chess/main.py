import gym
import gym_chess
import random
import wandb
import argparse
import torch
import torch.optim as optim
import secrets
from tqdm import tqdm
from alphazero import AlphaZeroNet, SelfPlay
from training import Trainer
from utils import (
    load_hyperparameters,
    ReplayBuffer,
    save_stats,
    plot_stats,
    ACTION_SIZE,
)


def main():
    config = load_hyperparameters()

    env = gym.make("Chess-v0")

    seed = secrets.randbelow(2**32)

    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    wandb.init(project="AlphaZero-Chess", name=f"Training_Seed_{seed}")
    wandb.config.update(config)

    # ensure the network output size matches our action mapping
    print(f"ACTION_SIZE = {ACTION_SIZE}")
    model = AlphaZeroNet(num_blocks=config["num_blocks"], num_actions=ACTION_SIZE)

    # Add small weight decay for L2 regularization (matches AlphaZero style)
    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler_T_max"], eta_min=config["scheduler_eta_min"]
    )
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_size"])

    trainer = Trainer(
        model=model,
        env=env,
        seed=seed,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval=config["scheduler_interval"],
        loss_fn=loss_fn,
        checkpoint_interval=config["checkpoint_interval"],
        device=device,
        replay_buffer=replay_buffer,
        batch_size=config["batch_size"],
        train_steps_per_iteration=config["train_steps_per_iteration"],
    )

    train_stats = {"avg_losses": [], "avg_value_losses": [], "avg_policy_losses": []}
    for iteration in tqdm(range(config["num_iterations"])):
        tqdm.write(f"Starting iteration {iteration + 1}/{config['num_iterations']}")
        self_play = SelfPlay(
            model,
            env,
            replay_buffer,
            mcts_simulations=config["mcts_simulations"],
            device=device,
        )

        training_data = []
        for _ in range(config["games_per_iteration"]):
            tqdm.write(f"Generating game {_ + 1}/{config['games_per_iteration']}")
            training_data.extend(self_play.generate_game())
            tqdm.write(f"Finished game {_ + 1}/{config['games_per_iteration']}")
        episode_stats = trainer.train(training_data)
        tqdm.write(f"Finished iteration {iteration + 1}/{config['num_iterations']}")
        train_stats["avg_losses"].append(episode_stats["avg_loss"])
        train_stats["avg_value_losses"].append(episode_stats["avg_value_loss"])
        train_stats["avg_policy_losses"].append(episode_stats["avg_policy_loss"])
    save_stats(train_stats)
    plot_stats(train_stats)


if __name__ == "__main__":
    main()
