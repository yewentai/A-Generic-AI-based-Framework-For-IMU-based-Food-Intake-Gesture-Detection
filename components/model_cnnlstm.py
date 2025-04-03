#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU CNN-LSTM Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script defines the CNN-LSTM architecture for IMU data classification.
              It employs dilated convolutions to preserve temporal resolution and processes
              the extracted features with an LSTM layer.
Reference   : A Data Driven End-to-End Approach for In-the-Wild Monitoring of Eating Behavior Using Smartwatches
Adjustment  : Use dilated convolutions instead of max pooling (which downsamples the time dimension) to expand the receptive field while preserving the temporal resolution.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv2 = nn.Conv1d(
            in_channels=f1, out_channels=f2, kernel_size=3, dilation=2, padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=f2, out_channels=f3, kernel_size=3, dilation=4, padding=4
        )
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


def CNNLSTM_Loss(outputs, targets):
    """
    Loss function for CNN_LSTM for multi-class classification.

    Assumes:
      - outputs: Tensor of shape [batch_size, num_classes, seq_len] (logits)
      - targets: Tensor of shape [batch_size, seq_len] with integer class labels
                 in the range [0, num_classes - 1].

    The loss is computed as a combination of:
      - Cross-Entropy loss (applied on each time step).
      - Temporal smoothing loss: MSE loss between the log-softmax predictions
        for consecutive time steps.

    Returns:
        ce_loss (torch.Tensor): Cross-entropy loss.
        mse_loss_mean (torch.Tensor): Temporal smoothing loss.
    """
    targets = targets.long()
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_mse_fn = nn.MSELoss(reduction="none")

    batch_size, num_classes, seq_len = outputs.size()

    # Compute cross-entropy loss.
    # Transpose outputs to [batch_size, seq_len, num_classes] and reshape.
    ce_loss = loss_ce_fn(
        outputs.transpose(2, 1).contiguous().view(-1, num_classes), targets.view(-1)
    )

    # Compute temporal smoothing loss if there is more than one time step.
    if seq_len > 1:
        mse_loss_value = loss_mse_fn(
            F.log_softmax(outputs[:, :, 1:], dim=1),
            F.log_softmax(outputs.detach()[:, :, :-1], dim=1),
        )
        mse_loss_value = torch.clamp(mse_loss_value, min=0, max=16)
        mse_loss_mean = torch.mean(mse_loss_value)
    else:
        mse_loss_mean = torch.tensor(0.0, device=outputs.device)

    return ce_loss, mse_loss_mean
