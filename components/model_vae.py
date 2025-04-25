#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-25
Description : This script defines the Variational Autoencoder (VAE) architecture for
              IMU data, including its encoder, decoder, and the combined VAE loss function.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

PATIENCE = 10


class VAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dim):
        super(VAE, self).__init__()
        self.sequence_length = sequence_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Automatically calculate feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            features = self.encoder(dummy)
            self.feature_dim = features.view(1, -1).size(-1)  # Automatically calculate feature dimensions

        # Fully connected layers
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.feature_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(z.size(0), 128, -1)  # 128 corresponds to the number of channels in the last encoder layer
        return self.decoder(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def VAE_Loss(recon, x, mu, logvar):
    """
    The VAE loss consists of the reconstruction loss and KL divergence.
    Here, Mean Squared Error (MSE) is used as the reconstruction loss.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


class VAEMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

        # Tracking variables
        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.lr_patience_counter = 0

    def log_metrics(self, epoch, recon_loss, kl_loss, lr, beta):
        """Log metrics to TensorBoard and print to console"""
        self.writer.add_scalar("Loss/recon", recon_loss, epoch)
        self.writer.add_scalar("Loss/kl", kl_loss, epoch)
        self.writer.add_scalar("Params/lr", lr, epoch)
        self.writer.add_scalar("Params/beta", beta, epoch)

        print(
            f"Epoch {epoch}: Recon Loss = {recon_loss:.4f}, KL Loss = {kl_loss:.4f}, LR = {lr:.2e}, Beta = {beta:.3f}"
        )

    def log_reconstructions(self, original, reconstructed, epoch, num_samples=5):
        """Log sample reconstructions"""
        with torch.no_grad():
            # Select random samples
            idx = torch.randperm(original.size(0))[:num_samples]
            orig = original[idx].cpu().numpy()
            recon = reconstructed[idx].cpu().numpy()

            # Plot each channel separately
            fig, axes = plt.subplots(6, num_samples, figsize=(15, 10))
            for i in range(num_samples):
                for j in range(6):  # 6 IMU channels
                    axes[j, i].plot(orig[i, j], "b-", label="Original")
                    axes[j, i].plot(recon[i, j], "r--", label="Recon")
                    axes[j, i].set_title(f"Ch{j+1}")
                    if i == 0 and j == 0:
                        axes[j, i].legend()

            plt.tight_layout()
            self.writer.add_figure("Reconstructions", fig, epoch)
            plt.close(fig)

    def check_early_stop(self, current_loss, lr):
        """Check if training should stop early"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            self.lr_patience_counter = 0
            return False
        else:
            self.epochs_no_improve += 1
            self.lr_patience_counter += 1
            if self.epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement")
                return True
            return False

    def close(self):
        self.writer.close()
