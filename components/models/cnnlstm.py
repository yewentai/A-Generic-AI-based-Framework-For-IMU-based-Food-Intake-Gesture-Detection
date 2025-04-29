#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU CNN-LSTM Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-29
Description : This script defines the CNN-LSTM architecture for IMU data classification.
              It employs dilated convolutions to preserve temporal resolution and processes
              the extracted features with an LSTM layer.
Reference   : A Data Driven End-to-End Approach for In-the-Wild Monitoring of Eating Behavior Using Smartwatches
Adjustment  : Use dilated convolutions instead of max pooling (which downsamples the time dimension) to expand the receptive field while preserving the temporal resolution.
===============================================================================
"""

import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM architecture using dilated convolutions to preserve the original
    temporal resolution (i.e. input (B,6,M) leads to output (B,N,M)), where N is
    the number of classes.

    The dilated convolutions increase the effective receptive field while
    keeping the output length the same as the input.

    Layer Details:
      - Conv1: kernel_size=5, dilation=1, padding=2  => effective kernel size = 5.
      - Conv2: kernel_size=3, dilation=2, padding=2  => effective kernel size = 5.
      - Conv3: kernel_size=3, dilation=4, padding=4  => effective kernel size = 9.
    """

    def __init__(
        self,
        input_channels=6,
        conv_filters=(32, 64, 128),
        lstm_hidden=128,
        num_classes=3,
    ):
        super(CNNLSTM, self).__init__()
        f1, f2, f3 = conv_filters

        # Convolutional layers with dilated convolutions
        # Use proper padding so that the output length remains M.
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=f1,
            kernel_size=5,
            dilation=1,
            padding=2,
        )
        self.conv2 = nn.Conv1d(in_channels=f1, out_channels=f2, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=f2, out_channels=f3, kernel_size=3, dilation=4, padding=4)
        self.relu = nn.ReLU()

        # LSTM that processes the sequence with length M
        self.lstm = nn.LSTM(input_size=f3, hidden_size=lstm_hidden, batch_first=True)

        # Fully connected layer to produce logits per time step for each class
        self.fc = nn.Linear(lstm_hidden, num_classes)
        # No sigmoid: CrossEntropyLoss expects raw logits.

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 6, M)

        Returns:
            out: Tensor of shape (B, N, M) where N is the number of classes.
        """
        # Apply dilated convolutions; output shape remains (B, *, M)
        x = self.relu(self.conv1(x))  # (B, 32, M)
        x = self.relu(self.conv2(x))  # (B, 64, M)
        x = self.relu(self.conv3(x))  # (B, 128, M)

        # Transpose to shape (B, M, 128) for the LSTM
        x = x.transpose(1, 2)

        # LSTM processing produces output of shape (B, M, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply the fully connected layer (on every time step)
        out = self.fc(lstm_out)  # (B, M, num_classes)
        # No activation applied because we'll use CrossEntropyLoss which
        # expects raw logits.

        # Transpose to get final output shape (B, num_classes, M)
        out = out.transpose(1, 2)
        return out
