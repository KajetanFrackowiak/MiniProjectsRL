import matplotlib.pyplot as plt
import yaml
import os


def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config

def plot(training_results, dir_name, file_name):
    epochs = range(1, len(training_results["avg_step_rewards"]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(epochs, training_results["avg_step_rewards"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Reward per Step")
    ax1.set_title("Step Rewards Over Training")

    ax2.plot(epochs, training_results["avg_policy_losses"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Policy Loss")
    ax2.set_title("Policy Loss")
    
    ax3.plot(epochs, training_results["avg_disc_losses"])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Discriminator Loss")
    ax3.set_title("Discriminator Loss")

    plt.tight_layout()

    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(f"{dir_name}/{file_name}.png")

    plt.show()