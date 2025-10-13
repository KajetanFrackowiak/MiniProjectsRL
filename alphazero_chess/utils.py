import yaml
import random
from collections import deque
import torch
import numpy as np
import json
import os
import matplotlib

matplotlib.use("Agg")  # use a non-interactive backend

import matplotlib.pyplot as plt


def load_hyperparameters(file_path="hyperparameters.yaml"):
    with open(file_path, "r") as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def board_to_tensor(board, device=None):
    """
    Convert a chess.Board object to a 12x8x8 tensor.
    6 piece types for each color -> 12 channels.
    """
    piece_map = board.piece_map()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Map piece symbols to channels
    piece_to_channel = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,  # white
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,  # black
    }

    for square, piece in piece_map.items():
        row = 7 - (square // 8)  # 0-indexed from top
        col = square % 8
        channel = piece_to_channel[piece.symbol()]
        tensor[channel, row, col] = 1.0

    t = torch.tensor(tensor)
    if device is not None:
        try:
            return t.to(device)
        except Exception:
            return t
    return t


def build_action_mappings():
    """
    Build a deterministic move (UCI) -> index and index -> move mapping.
    We'll use a simple encoding: all from-square (0..63) x to-square (0..63) = 4096
    plus promotion moves for pawn promotions (both colors), 4 promotion pieces each,
    producing 4096 + 512 = 4608 total actions. This is a simplified but stable mapping.
    Returns (action_to_index, index_to_action)
    """
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f + r for r in ranks for f in files]  # a1..h8

    index_to_action = []
    action_to_index = {}

    # base from->to moves
    for from_sq in squares:
        for to_sq in squares:
            uci = from_sq + to_sq
            action_to_index[uci] = len(index_to_action)
            index_to_action.append(uci)

    # add promotion moves: from rank 7 to rank 8 (white promotions), and from rank 2 to rank 1 (black)
    promo_pieces = ["q", "r", "b", "n"]
    # white promotions: from rank 7 (i.e., '7') to rank 8 ('8')
    for from_file in files:
        from_sq = from_file + "7"
        for to_file in files:
            to_sq = to_file + "8"
            for p in promo_pieces:
                uci = from_sq + to_sq + p
                action_to_index[uci] = len(index_to_action)
                index_to_action.append(uci)

    # black promotions: from rank 2 ('2') to rank 1 ('1')
    for from_file in files:
        from_sq = from_file + "2"
        for to_file in files:
            to_sq = to_file + "1"
            for p in promo_pieces:
                uci = from_sq + to_sq + p
                action_to_index[uci] = len(index_to_action)
                index_to_action.append(uci)

    return action_to_index, index_to_action


# build global mappings at import time
ACTION_TO_INDEX, INDEX_TO_ACTION = build_action_mappings()
ACTION_SIZE = len(INDEX_TO_ACTION)


def save_stats(stats, seed, dir_name="stats"):
    """
    Save training stats dict to JSON file named training_stats_{seed}.json
    """
    os.makedirs(dir_name, exist_ok=True)
    file_path = os.path.join(dir_name, f"training_stats_{seed}.json")
    with open(file_path, "w") as f:
        json.dump(stats, f)


def plot_stats(stats, seed):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(stats["avg_losses"], label="Average Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Average Total Loss")
    plt.legend()
    plt.savefig("plots/average_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats["avg_policy_losses"], label="Average Policy Loss", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Policy Loss")
    plt.title("Average Policy Loss")
    plt.legend()
    plt.savefig("plots/average_policy_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats["avg_value_losses"], label="Average Value Loss", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Value Loss")
    plt.title("Average Value Loss")
    plt.legend()
    plt.savefig(f"plots/average_value_loss_{seed}.png")
    plt.close()
