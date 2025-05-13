#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU AccNet Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-13
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
        input_channels: int = 3,
        conv_filters: tuple = (32, 64, 128),
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
    ):
        """
        Sequence-labeling AccNet with temporal upsampling:
        Processes accelerometer sequences and outputs per-timestep logits
        matching the original input length.
        """
        super(AccNet, self).__init__()

        # Convolutional blocks with pooling
        self.conv1 = nn.Conv1d(input_channels, conv_filters[0], kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(conv_filters[0])
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(conv_filters[1])
        self.conv3 = nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(conv_filters[2])

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)

        # 1x1 convolution to map features to class logits
        self.classifier = nn.Conv1d(conv_filters[2], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_channels, seq_len)
        Returns:
            out: Tensor of shape (B, num_classes, seq_len)
        """
        # Preserve original length
        original_len = x.size(2)

        # Conv + pool stages
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # length -> L/2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # length -> L/4
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # length -> L/8

        # Map to class logits at reduced resolution
        out = self.classifier(x)  # -> (B, num_classes, L/8)

        # Upsample logits back to original temporal resolution
        out = F.interpolate(out, size=original_len, mode="linear", align_corners=False)

        return out
