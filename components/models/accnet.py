#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU AccNet Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-28
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

import torch.nn as nn
import torch.nn.functional as F


class AccNet(nn.Module):
    def __init__(self, num_classes=3, input_channels=6):
        """
        Initialize the AccNet model.

        Parameters:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels (default is 3 for accelerometer data).
        """
        super(AccNet, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Max pooling to reduce temporal resolution
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Adaptive pooling to get fixed-size feature vector regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass for the AccNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, sequence_length).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        # Convolutional blocks with ReLU activation and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Shape: (B, 32, L/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Shape: (B, 64, L/4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Shape: (B, 128, L/8)

        # Adaptive pooling to aggregate temporal features
        x = self.adaptive_pool(x)  # Shape: (B, 128, 1)
        x = x.squeeze(-1)  # Shape: (B, 128)

        # Fully connected layer for final output
        x = self.fc(x)  # Shape: (B, num_classes)
        return x
