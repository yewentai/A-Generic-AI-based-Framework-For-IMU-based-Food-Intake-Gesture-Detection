#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU AccNet Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script defines the AccNet model for processing accelerometer
              data from IMU sensors. AccNet is designed as a lightweight convolutional
              network that serves as an encoder to extract compact feature representations
              from 1D accelerometer signals. The network consists of a series of 1D
              convolutional layers with batch normalization and ReLU activations, followed
              by max pooling layers to reduce temporal resolution. An adaptive pooling
              layer aggregates the features, which are then passed through a fully
              connected layer to produce the final class logits.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AccNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 6,
        conv_filters: tuple = (32, 64, 128),
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
        adaptive_pool_output_size: int = 1,
    ):
        """
        Initialize the configurable AccNet model.

        Parameters:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels.
            conv_filters (tuple): Number of filters for each conv layer.
            kernel_size (int): Convolutional kernel size (applied to all convs).
            padding (int): Padding size for convolutional layers.
            pool_kernel_size (int): Kernel size for max-pooling.
            adaptive_pool_output_size (int): Output size for adaptive pooling.
        """
        super(AccNet, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=conv_filters[0], kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(conv_filters[0])

        self.conv2 = nn.Conv1d(
            in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(conv_filters[1])

        self.conv3 = nn.Conv1d(
            in_channels=conv_filters[1], out_channels=conv_filters[2], kernel_size=kernel_size, padding=padding
        )
        self.bn3 = nn.BatchNorm1d(conv_filters[2])

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(adaptive_pool_output_size)

        # Fully connected classification layer
        self.fc = nn.Linear(conv_filters[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AccNet model.

        Args:
            x (torch.Tensor): Input of shape (B, input_channels, sequence_length).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (B, conv_filters[0], L/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (B, conv_filters[1], L/4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (B, conv_filters[2], L/8)

        x = self.adaptive_pool(x)  # -> (B, conv_filters[2], 1)
        x = x.squeeze(-1)  # -> (B, conv_filters[2])

        x = self.fc(x)  # -> (B, num_classes)
        return x
