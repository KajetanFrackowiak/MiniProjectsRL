import secrets
import argparse
import minari
import torch
import torch.optim as optim
from gail import PolicyNetwork, Discriminator
from expert import Expert
from training import Trainer
from utils import load_hyperparameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="mujoco/hopper/expert-v0")
    parser.add_argument("--train", action="store_true", help="Train the GAIL model")

    args = parser.parse_args()

    config = load_hyperparameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = minari.load_dataset(args.env, download=True)

    seed = secrets.randbelow(2**32)

    env = dataset.recover_environment() 
    env.reset(seed=seed)
    torch.manual_seed(seed)


    episodes = list(dataset.iterate_episodes())
    episode = episodes[0]

    expert = Expert(episode, device=device)

    policy_network = PolicyNetwork(
        env.observation_space.shape[0], env.action_space.shape[0], discrete=False
    ).to(device)
    discriminator = Discriminator(
        env.observation_space.shape[0], env.action_space.shape[0]
    ).to(device)

    policy_optimizer = optim.Adam(
        policy_network.parameters(), config["policy_learning_rate"]
    )
    policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=config["epochs"], eta_min=config["policy_eta_min"]
    )

    disc_optimizer = optim.Adam(
        discriminator.parameters(), config["disc_learning_rate"]
    )
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        disc_optimizer, T_max=config["epochs"], eta_min=config["disc_eta_min"]
    )


    trainer = Trainer(
        env=env,
        expert=expert,
        policy_network=policy_network,
        discriminator=discriminator,
        policy_optimizer=policy_optimizer,
        disc_optimizer=disc_optimizer,
        policy_scheduler=policy_scheduler,
        disc_scheduler=disc_scheduler,
        epochs=config["epochs"],
        episodes_per_epoch=config["episodes_per_epoch"],
        checkpoint_interval=config["checkpoint_interval"],
        device=device,
        seed=seed,
    )

    if args.train:
        training_metrics = trainer.train()

 

if __name__ == "__main__":
    main()
