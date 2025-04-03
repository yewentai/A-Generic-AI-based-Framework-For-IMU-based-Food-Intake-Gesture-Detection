#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script defines the Variational Autoencoder (VAE) architecture for
              IMU data, including its encoder, decoder, and the combined VAE loss function.
===============================================================================
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dim):
        """
        Parameters:
            input_channels (int): Number of input channels (e.g., 6 for IMU data).
            sequence_length (int): Length of the input sequence.
            hidden_dim (int): Hidden dimension (for extension purposes).
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        # Encoder: 1D convolutional layers to extract temporal features.
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the output length after convolutions.
        def conv_output_size(L, kernel_size=3, stride=2, padding=1):
            return (L + 2 * padding - kernel_size) // stride + 1

        l1 = conv_output_size(sequence_length, 3, 2, 1)
        l2 = conv_output_size(l1, 3, 2, 1)
        l3 = conv_output_size(l2, 3, 2, 1)
        self.conv_output_length = l3  # Number of time steps after the last conv layer.
        self.feature_dim = 128 * self.conv_output_length  # Flattened feature dimension.

        # Fully connected layers to generate latent mean and log-variance.
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder: Map the latent vector back to convolutional features and use transposed convolutions to reconstruct.
        self.decoder_input = nn.Linear(latent_dim, self.feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Use Sigmoid if data is normalized to [0, 1].
        )

    def encode(self, x):
        # x shape: (batch, channels, sequence_length)
        enc = self.encoder(x)  # (batch, 128, conv_output_length)
        enc_flat = enc.view(x.size(0), -1)  # Flatten to (batch, feature_dim)
        mu = self.fc_mu(enc_flat)
        logvar = self.fc_logvar(enc_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        dec_input = self.decoder_input(z)
        dec_input = dec_input.view(z.size(0), 128, self.conv_output_length)
        recon = self.decoder(dec_input)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def VAE_Loss(recon, x, mu, logvar):
    """
    The VAE loss consists of the reconstruction loss and KL divergence.
    Here, Mean Squared Error (MSE) is used as the reconstruction loss.
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
