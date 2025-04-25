#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Pre-Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-25
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
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from components.datasets import IMUDatasetX
from components.model_vae import VAE, VAE_Loss


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "dataset/Oreba/"  # Modify to your dataset directory
    X_PATH = os.path.join(data_dir, "X.pkl")

    with open(X_PATH, "rb") as f:
        X = np.array(pickle.load(f), dtype=object)

    sequence_length = 80  # Adjust as needed
    dataset = IMUDatasetX(X, sequence_length=sequence_length)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Model Construction ----------------------
    latent_dim = 64
    vae_model = VAE(input_channels=6, sequence_length=sequence_length, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
    num_epochs = 50

    vae_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{num_epochs}"):
            x = batch.to(device).permute(0, 2, 1)  # [B, channels, seq_len]
            optimizer.zero_grad()
            recon, mu, logvar = vae_model(x)
            loss = VAE_Loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # ---------------------- Save Pre-Trained VAE Model ----------------------
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    save_dir = os.path.join("result", version_prefix)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "pretrained_vae.pth")
    torch.save(vae_model.state_dict(), checkpoint_path)
    print(f"Pre-trained VAE model saved to {checkpoint_path}")

    config = {
        "model": "VAE",
        "input_channels": 6,
        "sequence_length": sequence_length,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 1e-3,
    }
    config_path = os.path.join(save_dir, "config_vae.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    main()
