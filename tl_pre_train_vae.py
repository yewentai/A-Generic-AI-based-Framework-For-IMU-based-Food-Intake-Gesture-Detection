#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Pre-Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-25
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
from components.model_vae import VAE, VAE_Loss, VAEMonitor

# Hyperparameters
BETA = 0.1
WARMUP_EPOCHS = 30
INITIAL_LR = 1e-3
MIN_LR = 1e-5
LR_PATIENCE = 5


def adjust_learning_rate(optimizer, current_lr, factor=0.5, min_lr=MIN_LR):
    """Reduce learning rate when loss plateaus"""
    new_lr = max(current_lr * factor, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "dataset/Oreba/"
    X_PATH = os.path.join(data_dir, "X.pkl")

    with open(X_PATH, "rb") as f:
        X = np.array(pickle.load(f), dtype=object)

    # Progressive training parameters
    initial_seq_len = 40  # Start with shorter sequences
    final_seq_len = 80  # Target sequence length
    seq_len_step = 20  # Increase length by this amount
    progressive_epochs = 15  # Train each sequence length for this many epochs

    # Initialize monitor
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    save_dir = os.path.join("result", version_prefix)
    monitor = VAEMonitor(save_dir)

    # ---------------------- Progressive Training Loop ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_lr = INITIAL_LR

    for seq_len in range(initial_seq_len, final_seq_len + 1, seq_len_step):
        print(f"\n=== Training with sequence length: {seq_len} ===")

        # Create dataset with current sequence length
        dataset = IMUDatasetX(X, sequence_length=seq_len)
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Initialize model (or reuse if continuing)
        if seq_len == initial_seq_len:
            latent_dim = 64
            vae_model = VAE(input_channels=6, sequence_length=seq_len, latent_dim=latent_dim).to(device)
            optimizer = optim.Adam(vae_model.parameters(), lr=current_lr)
        else:
            # Adjust model for new sequence length
            vae_model.sequence_length = seq_len
            vae_model = vae_model.to(device)

        # Train for progressive_epochs
        for epoch in range(progressive_epochs):
            vae_model.train()
            total_recon_loss = 0
            total_kl_loss = 0

            for batch in tqdm(dataloader, desc=f"SeqLen {seq_len} Epoch {epoch+1}/{progressive_epochs}"):
                x = batch.to(device).permute(0, 2, 1)  # [B, channels, seq_len]
                optimizer.zero_grad()
                recon, mu, logvar = vae_model(x)
                recon_loss, kl_loss = VAE_Loss(recon, x, mu, logvar)

                # KL warm-up strategy
                beta = min(BETA, BETA * epoch / WARMUP_EPOCHS)
                loss = recon_loss + beta * kl_loss

                loss.backward()
                optimizer.step()

                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

            # Calculate average losses
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)

            # Log metrics and reconstructions
            monitor.log_metrics(epoch, avg_recon_loss, avg_kl_loss, current_lr, beta)

            # Log sample reconstructions every 5 epochs
            if epoch % 5 == 0:
                monitor.log_reconstructions(x, recon, epoch)

            # Check for early stopping or learning rate adjustment
            if monitor.check_early_stop(avg_recon_loss, current_lr):
                break

            if monitor.lr_patience_counter >= LR_PATIENCE:
                current_lr = adjust_learning_rate(optimizer, current_lr)
                monitor.lr_patience_counter = 0
                print(f"Reduced learning rate to {current_lr:.2e}")

    # ---------------------- Save Final Model ----------------------
    checkpoint_path = os.path.join(save_dir, "pretrained_vae.pth")
    torch.save(vae_model.state_dict(), checkpoint_path)
    print(f"Final pre-trained VAE model saved to {checkpoint_path}")

    # Save config
    config = {
        "model": "VAE",
        "input_channels": 6,
        "final_sequence_length": final_seq_len,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "progressive_epochs": progressive_epochs,
        "final_learning_rate": current_lr,
        "beta": BETA,
        "warmup_epochs": WARMUP_EPOCHS,
    }
    config_path = os.path.join(save_dir, "config_vae.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    monitor.close()


if __name__ == "__main__":
    main()
