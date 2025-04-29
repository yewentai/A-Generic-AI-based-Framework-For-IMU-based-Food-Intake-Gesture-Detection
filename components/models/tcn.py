#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU TCN Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-29
Description : This script defines a Temporal Convolutional Network (TCN) model for
              IMU data processing, including dilated residual layers and a loss function.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual layer for the Temporal Convolutional Network (TCN).
    This layer applies a dilated convolution, followed by a ReLU activation,
    a 1x1 convolution, and dropout. It supports both causal and non-causal padding.

    Args:
        num_filters (int): Number of filters in the convolutional layers.
        dilation (int): Dilation factor for the dilated convolution.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        dropout (float): Dropout rate. Default is 0.3.
        causal (bool): Whether to use causal padding. Default is False.
    """

    def __init__(self, num_filters, dilation, kernel_size=3, dropout=0.3, causal=False):
        super().__init__()
        self.causal = causal
        if causal:
            # For causal padding, pad only on the left side
            self.pad = (kernel_size - 1) * dilation
            self.conv_dilated = nn.Conv1d(num_filters, num_filters, kernel_size, dilation=dilation)
        else:
            # For non-causal padding, pad symmetrically
            pad = dilation * (kernel_size - 1) // 2
            self.conv_dilated = nn.Conv1d(num_filters, num_filters, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()  # Activation function
        self.conv_1x1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)  # Pointwise convolution
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        """
        Forward pass of the dilated residual layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_filters, seq_len].

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        if self.causal:
            # Apply causal padding
            x_padded = F.pad(x, (self.pad, 0))
            out = self.conv_dilated(x_padded)
        else:
            # Non-causal convolution
            out = self.conv_dilated(x)
        out = self.relu(out)  # Apply ReLU activation
        out = self.conv_1x1(out)  # Apply 1x1 convolution
        out = self.dropout(out)  # Apply dropout
        return x + out  # Residual connection


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    Args:
        num_layers (int): Number of dilated residual layers.
        input_dim (int): Number of input features (channels).
        num_classes (int): Number of output classes.
        num_filters (int): Number of filters in each layer. Default is 128.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        dropout (float): Dropout rate for regularization. Default is 0.3.
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        num_classes,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super(TCN, self).__init__()
        # Initial 1x1 convolution to transform input to the desired number of filters
        self.conv_in = nn.Conv1d(input_dim, num_filters, kernel_size=1)

        # Stack of dilated residual layers
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(num_filters, dilation=2**i, kernel_size=kernel_size, dropout=dropout)
                for i in range(num_layers)
            ]
        )

        # Final 1x1 convolution to map to the desired number of output classes
        self.conv_out = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the TCN.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tensor: Output tensor of shape [batch_size, num_classes, seq_len].
        """
        # Apply initial 1x1 convolution
        out = self.conv_in(x)

        # Pass through each dilated residual layer
        for layer in self.layers:
            out = layer(out)

        # Apply final 1x1 convolution
        out = self.conv_out(out)

        return out


class MSTCN(nn.Module):
    def __init__(
        self,
        num_stages,
        num_layers,
        num_classes,
        input_dim,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super(MSTCN, self).__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList(
            [TCN(num_layers, input_dim, num_classes, num_filters, kernel_size, dropout) for _ in range(num_stages)]
        )
        self.conv_out = nn.Conv1d(num_classes * num_stages, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Forward pass of the MSTCN.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tensor: Output tensor of shape [batch_size, num_classes, seq_len].
        """
        # Apply each stage of TCN
        outputs = [stage(x) for stage in self.stages]

        # Concatenate outputs from all stages
        out = torch.cat(outputs, dim=1)

        # Apply final 1x1 convolution
        out = self.conv_out(out)

        return out


class MSTCN(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network.

    Args:
        num_stages   (int): Number of refinement stages.
        num_layers   (int): Number of dilated residual layers per stage.
        input_dim    (int): Number of input channels.
        num_classes  (int): Number of output classes.
        num_filters  (int): Hidden channel size in each stage.
        kernel_size  (int): Convolution kernel size for dilated layers.
        dropout      (float): Dropout rate inside each residual block.
        causal       (bool): If True use causal padding; else non-causal.
    """

    def __init__(
        self,
        num_stages,
        num_layers,
        input_dim,
        num_classes,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
        causal=False,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes

        # initial projection of raw input
        self.conv_in = nn.Conv1d(input_dim, num_filters, kernel_size=1)

        # build each stage's TCN trunk
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DilatedResidualLayer(
                            num_filters=num_filters,
                            dilation=2**i,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            causal=causal,
                        )
                        for i in range(num_layers)
                    ]
                )
                for _ in range(num_stages)
            ]
        )

        # head per stage: project hidden to class‐logits
        self.conv_outs = nn.ModuleList([nn.Conv1d(num_filters, num_classes, kernel_size=1) for _ in range(num_stages)])

        # inter‐stage projection: take logits → hidden
        self.conv_projs = nn.ModuleList(
            [nn.Conv1d(num_classes, num_filters, kernel_size=1) for _ in range(num_stages - 1)]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, input_dim, L]
        Returns:
            List[Tensor]: length=num_stages, each [B, num_classes, L]
        """
        # initial embedding
        hidden = self.conv_in(x)

        logits_list = []
        for stage_idx in range(self.num_stages):
            # pass through dilated residual layers
            for layer in self.stages[stage_idx]:
                hidden = layer(hidden)

            # project to class‐logits
            logits = self.conv_outs[stage_idx](hidden)
            logits_list.append(logits)

            # prepare input for next stage
            if stage_idx < self.num_stages - 1:
                hidden = self.conv_projs[stage_idx](logits)

        return logits_list


