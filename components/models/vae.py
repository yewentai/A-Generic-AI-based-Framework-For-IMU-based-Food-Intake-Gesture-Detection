#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU VAE Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script implements a Variational Autoencoder (VAE) for IMU data.
              It includes definitions for the encoder and decoder using 1D convolutional
              and transposed convolutional layers, the reparameterization trick, and
              a custom VAE loss function combining MSE reconstruction loss and KL divergence.
===============================================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.input_channels = input_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        # Calculate flattened size
        self.fc_input = 128 * (80 // (2**3))  # 80 input length reduced by 2^3 from 3 conv layers

        # Latent space
        self.fc_mu = nn.Linear(self.fc_input, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Decoder, self).__init__()

        # Decoder
        self.fc_input = 128 * (80 // (2**3))  # Match encoder's flattened size
        self.decoder_input = nn.Linear(latent_dim, self.fc_input)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), 128, -1)  # Reshape to (batch, 128, 10)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def VAE_Loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss
