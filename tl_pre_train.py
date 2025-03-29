#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Pre-Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 1.0
Created     : 2025-03-26
Description : This script pre-trains a Variational Autoencoder (VAE) on IMU (Inertial
              Measurement Unit) data in an unsupervised manner to extract latent features.
              The trained model weights and configuration are saved in a timestamped directory
              for subsequent downstream tasks.
===============================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim

from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring
from components.model_vae import VAE, VAE_Loss


# =============================================================================
#                         Configuration Parameters
# =============================================================================

# Dataset
DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
if DATASET.startswith("FD"):
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
elif DATASET.startswith("DX"):
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Dataloader
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
BATCH_SIZE = 64
NUM_WORKERS = 16
FLAG_DATASET_MIRROR = False

# VAE model hyperparameters
INPUT_DIM = 6
HIDDEN_DIM = 128
LATENT_DIM = 64

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4

# =============================================================================
#                       Data Loading
# =============================================================================
# Define file paths for the dataset
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Load left-hand and right-hand data from pickle files
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

if FLAG_DATASET_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge left and right hand data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the IMUDataset
full_dataset = IMUDataset(
    X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR
)
dataloader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- VAE Pre-Training --------------------
vae_model = VAE(
    input_channels=INPUT_DIM,
    sequence_length=WINDOW_SIZE,
    HIDDEN_DIM=HIDDEN_DIM,
    LATENT_DIM=LATENT_DIM,
).to(device)
optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
vae_model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_x, _ in tqdm(
        dataloader, desc=f"VAE Training Epoch {epoch+1}/{NUM_EPOCHS}"
    ):
        batch_x = batch_x.to(device)
        batch_x = batch_x.permute(
            0, 2, 1
        )  # Reshape to (batch, channels, sequence_length)
        optimizer.zero_grad()
        recon, mu, logvar = vae_model(batch_x)
        loss = VAE_Loss(recon, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader.dataset)

# -------------------- Save Model and Configuration --------------------
version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
result_dir = os.path.join("result", version_prefix)
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = os.path.join(result_dir, "pretrained_vae.pth")
torch.save(vae_model.state_dict(), checkpoint_path)

config = {
    "dataset": DATASET,
    "input_dim": INPUT_DIM,
    "HIDDEN_DIM": HIDDEN_DIM,
    "LATENT_DIM": LATENT_DIM,
    "sampling_freq": SAMPLING_FREQ,
    "window_size": WINDOW_SIZE,
    "batch_size": BATCH_SIZE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "downsample_factor": DOWNSAMPLE_FACTOR,
    "mirroring": FLAG_DATASET_MIRROR,
}
config_path = os.path.join(result_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
