import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import torch
import torch.nn.functional as F


def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config

