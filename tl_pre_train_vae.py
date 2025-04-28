#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Pre-Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-28
Description : This script pre-trains a Variational Autoencoder (VAE) on IMU
              data in an unsupervised manner to extract latent features.
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
from components.models.vae import VAE, VAE_Loss

# Hyperparameters
BETA = 0.1
WARMUP_EPOCHS = 30
INITIAL_LR = 1e-3
MIN_LR = 1e-5
LR_PATIENCE = 5
SEQ_LEN = 80  # Fixed sequence length


def adjust_learning_rate(optimizer, current_lr, factor=0.5, min_lr=MIN_LR):
    new_lr = max(current_lr * factor, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def main():
    data_dir = "dataset/Oreba/"
    X_PATH = os.path.join(data_dir, "X.pkl")

    with open(X_PATH, "rb") as f:
        X = np.array(pickle.load(f), dtype=object)

    # Initialize save directory
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    save_dir = os.path.join("result", version_prefix)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = IMUDatasetX(X, sequence_length=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Model
    latent_dim = 64
    vae_model = VAE(input_channels=6, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=INITIAL_LR)

    current_lr = INITIAL_LR
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(100):
        vae_model.train()
        total_recon_loss = 0
        total_kl_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = batch.to(device).permute(0, 2, 1)
            optimizer.zero_grad()
            recon, mu, logvar = vae_model(x)
            recon_loss, kl_loss = VAE_Loss(recon, x, mu, logvar)

            # KL warm-up
            beta = min(BETA, BETA * epoch / WARMUP_EPOCHS)
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Beta: {beta:.4f}")

        # Learning rate adjustment
        if avg_recon_loss > best_loss:
            patience_counter += 1
            if patience_counter >= LR_PATIENCE:
                current_lr = adjust_learning_rate(optimizer, current_lr)
                print(f"LR reduced to {current_lr:.2e}")
                patience_counter = 0
        else:
            best_loss = avg_recon_loss
            patience_counter = 0

    # Save model
    checkpoint_path = os.path.join(save_dir, "pretrained_vae.pth")
    torch.save(vae_model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Save config
    config = {
        "model": "VAE",
        "input_channels": 6,
        "latent_dim": latent_dim,
        "sequence_length": SEQ_LEN,
        "initial_lr": INITIAL_LR,
        "beta": BETA,
        "warmup_epochs": WARMUP_EPOCHS,
    }
    with open(os.path.join(save_dir, "config_vae.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
