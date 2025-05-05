import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import MinMaxScaler
import os
import wandb

# --- Seed setting for reproducibility ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# --- Training Configuration ---
RAE_CONFIG = {
    "learning_rate": 0.0003,          # slightly lower for stability
    "dropout": 0.3,                  
    "batch_size": 64,
    "epochs": 20,                     # a bit more training
    "sequence_length": 15,           
    "num_layers": 3,                 
    "hidden_size": 128,              
    "weight_decay": 1e-5
}

# --- Sequence preparation ---
def create_sequences(data, seq_length):
    """Creates sequences from the data."""
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

# --- Reconstruction error calculation ---
def reconstruction_errors(model, dataloader, device):
    """Calculates per-sample reconstruction error for sequential data."""
    model.to(device)
    model.eval()
    errors = []

    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            reconstructions = model(sequences)

            mse = torch.mean((sequences - reconstructions) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())

    return np.array(errors)

