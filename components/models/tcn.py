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


def TCN_Loss(outputs, targets, smoothing="mse", max_diff=16.0):
    """
    Compute TCN loss: cross-entropy + various temporal smoothing penalties.

    Args:
        outputs (Tensor): logits of shape [B, C, L].
        targets (Tensor): labels of shape [B, L], long, with ignore_index=-100.
        smoothing (str): one of ['mse','l1','huber','kl','js','tv','sec_diff','emd','spectral'].
        max_diff (float): clamp threshold for log-prob differences (only for diff-based).
    Returns:
        ce_loss (Tensor): the CrossEntropyLoss.
        smooth_loss (Tensor): the chosen smoothing loss.
    """
    # --- Cross‐Entropy Part ---
    targets = targets.long()
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    B, C, L = outputs.size()
    logits_flat = outputs.transpose(2, 1).contiguous().view(-1, C)
    targets_flat = targets.view(-1)
    ce_loss = ce_fn(logits_flat, targets_flat)

    # If seq_len == 1, skip smoothing
    if L <= 1:
        return ce_loss, torch.tensor(0.0, device=outputs.device)

    # Precompute log-probs and shifted versions
    logp = F.log_softmax(outputs, dim=1)  # [B,C,L]
    log_prev = logp[:, :, :-1].detach()  # stop grad beyond prev
    log_next = logp[:, :, 1:]
    diff = log_next - log_prev  # [B,C,L-1]

    # Select smoothing
    if smoothing == "mse":
        smooth_loss = diff.clamp(-max_diff, max_diff).pow(2).mean()

    elif smoothing == "l1":
        smooth_loss = diff.clamp(-max_diff, max_diff).abs().mean()

    elif smoothing == "huber":
        # smooth_l1: L2 for small, L1 for large
        smooth_loss = F.smooth_l1_loss(log_next, log_prev, reduction="mean")

    elif smoothing == "kl":
        # KL( P_{t+1} || P_t )
        smooth_loss = F.kl_div(log_next, log_prev.exp(), reduction="batchmean")

    elif smoothing == "js":
        # Jensen–Shannon divergence
        p_next = log_next.exp()
        p_prev = log_prev.exp()
        m = 0.5 * (p_next + p_prev)
        # KL(P||M) + KL(Q||M)
        smooth_loss = 0.5 * (
            F.kl_div(log_next, m.detach(), reduction="batchmean")
            + F.kl_div(log_prev, m.detach(), reduction="batchmean")
        )

    elif smoothing == "tv":
        # Total Variation: sum over classes, mean over batch+time
        smooth_loss = diff.abs().sum(dim=1).mean()

    elif smoothing == "sec_diff":
        # Second‐order difference: penalize change of slope
        d1 = diff  # [B,C,L-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]  # [B,C,L-2]
        smooth_loss = d2.pow(2).mean()

    elif smoothing == "emd":
        # 1D EMD via L1 of CDF differences
        p_next = log_next.exp()
        p_prev = log_prev.exp()
        cdf_next = torch.cumsum(p_next, dim=1)
        cdf_prev = torch.cumsum(p_prev, dim=1)
        smooth_loss = (cdf_next - cdf_prev).abs().mean()

    elif smoothing == "spectral":
        # Spectral high‐freq penalty: mean power of non‐DC freqs
        fft = torch.fft.rfft(logp, dim=2)
        mag2 = fft[..., 1:].abs().pow(2)
        smooth_loss = mag2.mean()

    else:
        raise ValueError(f"Unknown smoothing type '{smoothing}'")

    return ce_loss, smooth_loss


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


def MSTCN_Loss(logits_list, targets, smoothing="mse", max_diff=16.0):
    """
    Compute deep‐supervised loss for MSTCN.

    Args:
        logits_list   (List[Tensor]): list of stage‐outputs, each [B,C,L].
        targets       (Tensor): [B,L] long with ignore_index=-100.
        smoothing     (str): which smoothing penalty to apply.
        max_diff      (float): clamp threshold for diff‐based.
        ce_weight     (float): weight for each stage's CE loss.
        smooth_weight (float): weight for each stage's smooth loss.

    Returns:
        Tuple(Tensor, Tensor): total_ce_loss, total_smooth_loss
    """
    total_ce = 0.0
    total_smooth = 0.0

    for logits in logits_list:
        ce, smooth = TCN_Loss(outputs=logits, targets=targets, smoothing=smoothing, max_diff=max_diff)
        total_ce += ce
        total_smooth += smooth

    return total_ce, total_smooth
