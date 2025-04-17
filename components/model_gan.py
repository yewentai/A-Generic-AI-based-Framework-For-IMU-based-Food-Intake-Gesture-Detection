#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================================
Event-based Segmentation GAN Model Definition
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-17
Description : This module defines the Generator and Discriminator classes for a GAN model
              used in augmenting event-based IMU segmentation data.
================================================================================================
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=6, target_length=240):
        super(Generator, self).__init__()

        self.target_length = target_length
        self.output_channels = output_channels

        # Starting with a latent vector and projecting it to a 1D sequence
        self.initial_linear = nn.Linear(latent_dim, 512 * 15)  # Project to initial sequence length

        # Transposed convolution layers to progressively increase sequence length
        self.main = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Using Tanh for normalized output
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        x = self.initial_linear(z)
        x = x.view(-1, 512, 15)  # Reshape to (batch_size, 512, 15)
        x = self.main(x)  # Shape becomes (batch_size, output_channels, 240)

        # Ensure output length matches target_length
        seq_len = x.size(2)
        if seq_len > self.target_length:
            x = x[:, :, : self.target_length]
        elif seq_len < self.target_length:
            pad = self.target_length - seq_len
            x = nn.functional.pad(x, (0, pad))

        return x  # Shape: (batch_size, output_channels, target_length)


class Discriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5 - Output layer
            nn.Conv1d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        x = self.main(x)
        return x.view(-1)  # Flatten to (batch_size,)
